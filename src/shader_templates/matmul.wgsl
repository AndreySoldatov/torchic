struct Params {
  M: u32,
  N: u32,
  K: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>; // MxK
@group(0) @binding(1) var<storage, read> B: array<f32>; // KxN
@group(0) @binding(2) var<storage, read_write> C: array<f32>; // MxN
@group(0) @binding(3) var<storage, read> p: Params;

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;

const TM: u32 = 4u;
const TN: u32 = 4u;

const BM: u32 = WG_Y * TM;
const BN: u32 = WG_X * TN;
const BK: u32 = 16u;

var<workgroup> As: array<f32, BM * BK>;
var<workgroup> Bs: array<f32, BK * BN>;

fn a_index(r: u32, c: u32) -> u32 {
  return r * p.K + c;
}

fn b_index(r: u32, c: u32) -> u32 {
  return r * p.N + c;
}

fn c_index(r: u32, c: u32) -> u32 {
  return r * p.N + c;
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let local_linear = lid.y * WG_X + lid.x;

  let block_row = wg.y * BM;
  let block_col = wg.x * BN;

  let thread_row_base = lid.y * TM;
  let thread_col_base = lid.x * TN;

  var acc: array<array<f32, TN>, TM>;
  for (var i = 0u; i < TM; i++) {
    for (var j = 0u; j < TN; j++) {
      acc[i][j] = 0.0;
    }
  }

  let num_a_loads = (BM * BK) / (WG_X * WG_Y);
  let num_b_loads = (BK * BN) / (WG_X * WG_Y);

  for (var k0 = 0u; k0 < p.K; k0 += BK) {
    for (var t = 0u; t < num_a_loads; t++) {
      let idx = local_linear + t * (WG_X * WG_Y);
      let r = idx / BK;
      let c = idx % BK;

      let gr = block_row + r;
      let gc = k0 + c;

      As[idx] = select(0.0, A[a_index(gr, gc)], gr < p.M && gc < p.K);
    }

    for (var t = 0u; t < num_b_loads; t++) {
      let idx = local_linear + t * (WG_X * WG_Y);
      let r = idx / BN;
      let c = idx % BN;

      let gr = k0 + r;
      let gc = block_col + c;

      Bs[idx] = select(0.0, B[b_index(gr, gc)], gr < p.K && gc < p.N);
    }

    workgroupBarrier();

    for (var kk = 0u; kk < BK; kk++) {
      var a_frag: array<f32, TM>;
      var b_frag: array<f32, TN>;

      for (var i = 0u; i < TM; i++) {
        let r = thread_row_base + i;
        a_frag[i] = As[r * BK + kk];
      }

      for (var j = 0u; j < TN; j++) {
        let c = thread_col_base + j;
        b_frag[j] = Bs[kk * BN + c];
      }

      for (var i = 0u; i < TM; i++) {
        for (var j = 0u; j < TN; j++) {
          acc[i][j] = fma(a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var i = 0u; i < TM; i++) {
    let gr = block_row + thread_row_base + i;
    if (gr >= p.M) { continue; }

    for (var j = 0u; j < TN; j++) {
      let gc = block_col + thread_col_base + j;
      if (gc >= p.N) { continue; }
      C[c_index(gr, gc)] = acc[i][j];
    }
  }
}