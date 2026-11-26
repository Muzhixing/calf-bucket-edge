// Fixed-model RKNN bridge. GridSample uses bilinear, zeros, align_corners=true,
// exactly the two operators in the hash-bound Fast-ACVNet export.
#include "rknn_api.h"
#include "rknn_custom_op.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool floating(rknn_tensor_type t) {
  return t == RKNN_TENSOR_FLOAT16 || t == RKNN_TENSOR_FLOAT32;
}
static float read_value(const rknn_custom_op_tensor &t, size_t i) {
  return t.attr.type == RKNN_TENSOR_FLOAT32
      ? static_cast<const float *>(t.mem.virt_addr)[i]
      : static_cast<float>(static_cast<const __fp16 *>(t.mem.virt_addr)[i]);
}
static void write_value(rknn_custom_op_tensor &t, size_t i, float value) {
  if (t.attr.type == RKNN_TENSOR_FLOAT32) static_cast<float *>(t.mem.virt_addr)[i] = value;
  else static_cast<__fp16 *>(t.mem.virt_addr)[i] = static_cast<__fp16>(value);
}
static int grid(rknn_custom_op_context *, rknn_custom_op_tensor *in, uint32_t ni,
                rknn_custom_op_tensor *out, uint32_t no) {
  if (ni != 2 || no != 1) return -1;
  const auto &x = in[0].attr; const auto &g = in[1].attr; const auto &y = out[0].attr;
  if (x.n_dims != 4 || g.n_dims != 4 || y.n_dims != 4 || x.dims[0] != 1 ||
      g.dims[0] != 1 || g.dims[3] != 2 || y.dims[0] != 1 ||
      y.dims[1] != x.dims[1] || y.dims[2] != g.dims[1] || y.dims[3] != g.dims[2] ||
      x.fmt == RKNN_TENSOR_NHWC || y.fmt == RKNN_TENSOR_NHWC ||
      !floating(x.type) || !floating(g.type) || !floating(y.type)) {
    fprintf(stderr, "unsupported_grid_tensor_layout x=%u,%u,%u,%u fmt=%d g=%u,%u,%u,%u y=%u,%u,%u,%u fmt=%d\n",
      x.dims[0],x.dims[1],x.dims[2],x.dims[3],x.fmt,g.dims[0],g.dims[1],g.dims[2],g.dims[3],
      y.dims[0],y.dims[1],y.dims[2],y.dims[3],y.fmt);
    return -1;
  }
  for (auto *t : {&in[0], &in[1], &out[0]}) {
    size_t bytes = t->attr.n_elems * (t->attr.type == RKNN_TENSOR_FLOAT16 ? 2u : 4u);
    if (!t->mem.virt_addr || t->mem.size < bytes || t->mem.offset != 0 ||
        (t->attr.size_with_stride && t->attr.size_with_stride != bytes)) return -1;
  }
  const int C=x.dims[1], H=x.dims[2], W=x.dims[3], OH=g.dims[1], OW=g.dims[2];
  // Reject other models/layouts instead of silently applying the fixed export semantics.
  if (H != 96 || W != 160 || OW != 160 || !((C == 48 && OH == 480) || (C == 16 && OH == 2304))) return -1;
  std::vector<int> xs(OH*OW), ys(OH*OW);
  std::vector<float> wx(OH*OW), wy(OH*OW);
  for (int i=0; i<OH*OW; ++i) {
    const float gx=read_value(in[1],2*i), gy=read_value(in[1],2*i+1);
    if (!std::isfinite(gx) || !std::isfinite(gy)) return -1;
    const float px=(gx+1)*.5f*(W-1), py=(gy+1)*.5f*(H-1);
    if (px < -2 || px > W+1 || py < -2 || py > H+1) {
      xs[i]=-2; ys[i]=-2; wx[i]=wy[i]=0; continue;
    }
    xs[i]=static_cast<int>(std::floor(px)); ys[i]=static_cast<int>(std::floor(py));
    wx[i]=px-xs[i]; wy[i]=py-ys[i];
  }
  #pragma omp parallel for num_threads(4)
  for (int c=0; c<C; ++c) {
    for (int i=0; i<OH*OW; ++i) {
      auto at = [&](int yy, int xx) -> float {
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) return 0.f;
        return read_value(in[0], (c*H+yy)*W+xx);
      };
      const int xx=xs[i], yy=ys[i]; const float u=wx[i], v=wy[i];
      const float value=at(yy,xx)*(1-u)*(1-v) + at(yy,xx+1)*u*(1-v)
                       + at(yy+1,xx)*(1-u)*v + at(yy+1,xx+1)*u*v;
      write_value(out[0], c*OH*OW+i, value);
    }
  }
  return 0;
}

struct Model { rknn_context ctx=0; };
extern "C" void *acv_open(const char *path) {
  auto *model = new Model();
  if (rknn_init(&model->ctx, const_cast<char *>(path), 0, 0, nullptr) != 0) { delete model; return nullptr; }
  if (const char *core = std::getenv("BUCKET_ACV_NPU_CORE")) {
    if (std::strlen(core)!=1 || core[0]<'0' || core[0]>'2' ||
        rknn_set_core_mask(model->ctx,static_cast<rknn_core_mask>(1 << (core[0]-'0')))) {
      rknn_destroy(model->ctx); delete model; return nullptr;
    }
    fprintf(stderr,"ASSIST_NPU_CORE=%s\n",core);
  }
  rknn_sdk_version v{};
  if (!rknn_query(model->ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)))
    fprintf(stderr,"ASSIST_RUNTIME api=%s driver=%s\n",v.api_version,v.drv_version);
  rknn_custom_op op{}; op.version=1; op.target=RKNN_TARGET_TYPE_CPU;
  std::strcpy(op.op_type,"GridSample"); op.compute=grid;
  if (rknn_register_custom_ops(model->ctx,&op,1) != 0) {
    rknn_destroy(model->ctx); delete model; return nullptr;
  }
  rknn_input_output_num io{};
  if (rknn_query(model->ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)) || io.n_input!=2 || io.n_output!=1) {
    rknn_destroy(model->ctx); delete model; return nullptr;
  }
  return model;
}
extern "C" int acv_infer(void *handle, float *left, float *right, float *result) {
  if (!handle || !left || !right || !result) return -1;
  auto *model = static_cast<Model *>(handle);
  rknn_input in[2]{};
  for (int i=0;i<2;++i) {
    in[i].index=i; in[i].buf=i?right:left; in[i].size=1*3*384*640*sizeof(float);
    in[i].type=RKNN_TENSOR_FLOAT32; in[i].fmt=RKNN_TENSOR_NCHW;
  }
  int rc=rknn_inputs_set(model->ctx,2,in); if (rc) return rc;
  rc=rknn_run(model->ctx,nullptr); if (rc) return rc;
  rknn_output output{}; output.want_float=1;
  rc=rknn_outputs_get(model->ctx,1,&output,nullptr); if (rc) return rc;
  if (output.size != 384*640*sizeof(float) || !output.buf) rc=-1;
  else std::memcpy(result,output.buf,output.size);
  rknn_outputs_release(model->ctx,1,&output);
  return rc;
}
extern "C" void acv_close(void *handle) {
  if (!handle) return;
  auto *model=static_cast<Model *>(handle); rknn_destroy(model->ctx); delete model;
}

// Boundary/numerical check without RKNN/NPU or any device I/O.
extern "C" float acv_grid_boundary_check() {
  const int C=48,H=96,W=160,OH=480,OW=160;
  std::vector<float> x(C*H*W), g(OH*OW*2), y(C*OH*OW);
  for (int c=0;c<C;++c) for (int h=0;h<H;++h) for(int w=0;w<W;++w)
    x[(c*H+h)*W+w]=1.f+w+10*h;
  const float coords[][3]={{0,0,1},{159,95,1110},{20.5f,30.25f,324},
                           {-.5f,10,50.5f},{160,10,0},{89.25f,10.5f,195.25f}};
  for(int i=0;i<OH*OW;++i) {
    g[2*i]=coords[i%6][0]*2/(W-1)-1;
    g[2*i+1]=coords[i%6][1]*2/(H-1)-1;
  }
  rknn_custom_op_tensor in[2]{}, out[1]{};
  auto setup=[](rknn_custom_op_tensor &t, float *p, uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    t.attr.n_dims=4; t.attr.dims[0]=n;t.attr.dims[1]=c;t.attr.dims[2]=h;t.attr.dims[3]=w;
    t.attr.n_elems=n*c*h*w;t.attr.type=RKNN_TENSOR_FLOAT32;t.attr.fmt=RKNN_TENSOR_NCHW;
    t.mem.virt_addr=p;t.mem.size=n*c*h*w*sizeof(float);
  };
  setup(in[0],x.data(),1,C,H,W);setup(in[1],g.data(),1,OH,OW,2);setup(out[0],y.data(),1,C,OH,OW);
  if(grid(nullptr,in,2,out,1))return -1;
  float error=0;
  for(int c=0;c<C;++c)for(int i=0;i<OH*OW;++i)
    error=std::fmax(error,std::fabs(y[c*OH*OW+i]-coords[i%6][2]));
  return error;
}
