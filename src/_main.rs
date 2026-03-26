use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    sync::{LazyLock, Mutex},
    vec,
};

use slotmap::{SecondaryMap, SlotMap, new_key_type};

new_key_type! {
    struct TNodeId;
}

struct TDagStore {
    nodes: SlotMap<TNodeId, TNode>,

    // This flag is set when any of nodes in the map loses it's last external owner
    dirty: bool,
}

impl TDagStore {
    // Returns list of all nodes that still have existing external reference
    fn roots(&self) -> Vec<TNodeId> {
        self.nodes
            .iter()
            .filter(|(_, v)| v.rc > 0)
            .map(|(k, _)| k)
            .collect()
    }

    // Expensive computation that traverses nodes graph and deletes the ones that ar
    fn cleanup(&mut self) {
        if !self.dirty {
            return;
        }

        let mut alive: HashSet<TNodeId> = HashSet::new();
        let mut stack = VecDeque::from_iter(self.roots());

        while let Some(id) = stack.pop_back() {
            if !alive.insert(id) {
                continue;
            }

            let node = self
                .nodes
                .get(id)
                .expect("Node should exist at this point. If it does not it's a bug");

            for parent in &node.parents {
                stack.push_back(*parent);
            }
        }

        for k in self.nodes.keys().collect::<Vec<_>>() {
            if !alive.contains(&k) {
                self.nodes.remove(k);
            }
        }

        self.dirty = false;
    }

    fn debug_dump_nodes(&self) {
        for (k, v) in &self.nodes {
            println!("{:?}: {}", k.0, v);
        }
    }
}

static TDAG_STORE: LazyLock<Mutex<TDagStore>> = LazyLock::new(|| {
    Mutex::new(TDagStore {
        nodes: SlotMap::with_key(),
        // rev: SecondaryMap::new(),
        dirty: false,
    })
});

#[derive(Debug, Clone)]
enum Op {
    Buffer(BufferId),
    Add,
}

struct TNode {
    parents: Vec<TNodeId>,
    op: Op,
    shape: Vec<usize>,

    // External reference counter that manages the root/non-root tensor separation
    // It DOES NOT show whether tensor is marked for deletion or not. It mereley shows whether someone is owning
    // this node externally. Node is deleted if a node doesn't have a root parent (may be transitive)
    rc: u32,
}

impl Display for TNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[rc: {}; op: {:?}]", self.rc, self.op)
    }
}

struct Tensor(TNodeId);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        TDAG_STORE
            .lock()
            .unwrap()
            .nodes
            .get_mut(self.0)
            .expect("The node is expected to exist during the copy")
            .rc += 1;

        Self(self.0)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let mut dag_lock = TDAG_STORE.lock().unwrap();

        let node = dag_lock
            .nodes
            .get_mut(self.0)
            .expect("The node is expected to exist during drop");

        assert!(
            node.rc > 0,
            "Tensor rc must be greater then zero. If not it's use after free and something broke"
        );
        node.rc -= 1;

        if node.rc == 0 {
            dag_lock.dirty = true;
        }
    }
}

impl Tensor {
    fn new(shape: Vec<usize>) -> Self {
        let bsize = shape.iter().product();
        let tnode = TNode {
            op: Op::Buffer(BUFFER_ALLOC.lock().unwrap().request(bsize)),
            parents: vec![],
            shape: shape,

            rc: 1,
        };

        let node = TDAG_STORE.lock().unwrap().nodes.insert(tnode);
        Self(node)
    }

    fn add(&self, other: &Self) -> Self {
        let tnode = TNode {
            parents: vec![self.0, other.0],
            op: Op::Add,
            shape: TDAG_STORE
                .lock()
                .unwrap()
                .nodes
                .get(self.0)
                .unwrap()
                .shape
                .clone(),

            rc: 1,
        };

        let node = TDAG_STORE.lock().unwrap().nodes.insert(tnode);

        Self(node)
    }

    fn realize(&self) -> Self {
        let shape = TDAG_STORE
            .lock()
            .unwrap()
            .nodes
            .get(self.0)
            .unwrap()
            .shape
            .clone();
        let bsize = shape.iter().product();

        let tnode = TNode {
            parents: vec![self.0],
            op: Op::Buffer(BUFFER_ALLOC.lock().unwrap().request(bsize)),
            shape: shape,

            rc: 1,
        };

        let node = TDAG_STORE.lock().unwrap().nodes.insert(tnode);

        schedule(node);

        Self(node)
    }
}

new_key_type! {
    struct BufferId;
}

struct Buffer {
    size: usize,
    in_use: bool,
}

struct BufferAllocator {
    pool: SlotMap<BufferId, Buffer>,
}

impl BufferAllocator {
    fn new() -> Self {
        Self {
            pool: SlotMap::with_key(),
        }
    }

    fn request(&mut self, size: usize) -> BufferId {
        if let Some(id) = self
            .pool
            .iter()
            .find(|(_, b)| b.size == size && !b.in_use)
            .map(|(id, _)| id)
        {
            let buf = self.pool.get_mut(id).unwrap();
            buf.in_use = true;
            buf.size = size;

            return id;
        } else {
            let buf = Buffer {
                in_use: true,
                size: size,
            };

            return self.pool.insert(buf);
        }
    }

    fn free(&mut self, buf: BufferId) {
        self.pool.get_mut(buf).unwrap().in_use = false;
    }
}

static BUFFER_ALLOC: LazyLock<Mutex<BufferAllocator>> =
    LazyLock::new(|| Mutex::new(BufferAllocator::new()));

fn topo_dfs(node: TNodeId, order: &mut Vec<TNodeId>, visited: &mut HashSet<TNodeId>) {
    if visited.contains(&node) {
        return;
    }

    visited.insert(node);

    let parents = {
        TDAG_STORE
            .lock()
            .unwrap()
            .nodes
            .get(node)
            .unwrap()
            .parents
            .clone()
    };

    for parent in parents {
        topo_dfs(parent, order, visited);
    }

    order.push(node);
}

fn topo(root: TNodeId) -> Vec<TNodeId> {
    let mut visited: HashSet<TNodeId> = HashSet::new();
    let mut order: Vec<TNodeId> = vec![];

    topo_dfs(root, &mut order, &mut visited);

    order
}

struct ExecItem {
    ast: Vec<TNodeId>,
    inputs: Vec<BufferId>,
    outputs: Vec<BufferId>,
}

fn saturate_subgraph(root: TNodeId, subgrapg: &mut HashSet<TNodeId>) {
    //check here:
}

fn schedule(tensor: TNodeId) -> Vec<ExecItem> {
    let rev_order = {
        let mut order = topo(tensor);
        order.reverse();
        order
    };

    // for n in &rev_order {
    //     println!("{:?}: {}", n.0, TDAG_STORE.lock().unwrap().nodes[*n]);
    // }
    // println!();

    let mut covered: HashSet<TNodeId> = HashSet::new();
    let mut groups: Vec<HashSet<TNodeId>> = vec![];

    for n in rev_order {
        if covered.contains(&n) {
            continue;
        }
    }

    vec![]
}

fn main() {
    let mut alive_test = Tensor::new(vec![]);

    for _ in 0..2 {
        let a = Tensor::new(vec![12, 4]);
        let b = Tensor::new(vec![12, 4]);

        alive_test = a.add(&alive_test);

        let r = a.add(&b).add(&a).add(&b).realize();
    }

    TDAG_STORE.lock().unwrap().cleanup();

    // TDAG_STORE.lock().unwrap().debug_dump_nodes();
}
