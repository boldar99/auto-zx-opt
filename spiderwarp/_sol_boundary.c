/* Selene 0.3 ABI adapter. Narsil owns scheduling; PECOS owns all noisy physics.
 * Reserved sleep calls are zero-duration phase barriers, never real sleeps.
 * At an ideal boundary we still advance PECOS's clocks using a sink simulator,
 * then execute it through a zero-scale PECOS instance on the real simulator.
 * The zero-scale instance also interprets Narsil's opaque QMC operations. This needs
 * leak2depolar=true and coherent_dephasing=false: no hidden leakage/coherent
 * state may survive an ideal region. The Python worker enforces both settings.
 */
#include <selene/error_model.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#define BEGIN_NOISE UINT64_C(1152921504606846977)
#define END_NOISE   UINT64_C(1152921504606846978)

/* Selene runs one emulator per subprocess. Both descriptors in this library
 * share this state. Reject multiple live runtime instances rather than aliasing
 * them. Selene's n_processes uses independent subprocesses, not threads. */
static int runtime_instances = 0, model_instances = 0;
static int phase = 0, pending_phase = -1, switches = 0;
static uint64_t live = 0, peak = 0, noisy_batches = 0, ideal_batches = 0;
static uint64_t noisy_rzz = 0, noisy_rxy = 0, noisy_measure = 0;
static uint64_t noisy_start = UINT64_MAX, noisy_end = 0;

typedef struct {
    void *library, *inner;
    SeleneRuntimePluginDescriptorV1 api;
    uint64_t limit;
} Runtime;
typedef struct {
    void *library, *inner, *ideal;
    SeleneErrorModelPluginDescriptorV1 api;
} Model;

static void *descriptor(void *lib, const char *symbol, const char *accessor) {
    void *p = dlsym(lib, symbol);
    if (!p) {
        const void *(*get)(void) = dlsym(lib, accessor);
        if (get) p = (void *)get();
    }
    return p;
}

static SeleneErrno rt_init(void **handle, uint64_t n, uint64_t start,
                           uint32_t argc, const char *const *argv) {
    if (runtime_instances || argc < 2) return -1;
    Runtime *r = calloc(1, sizeof(*r));
    if (!r) return -1;
    r->limit = strtoull(argv[1], NULL, 10);
    r->library = dlopen(argv[0], RTLD_NOW | RTLD_LOCAL);
    if (!r->library) { fprintf(stderr, "%s\n", dlerror()); free(r); return -1; }
    SeleneRuntimePluginDescriptorV1 *p = descriptor(r->library,
        "selene_runtime_plugin_descriptor_v1", "selene_runtime_get_plugin_descriptor_v1");
    if (!p || p->struct_size < sizeof(*p) || p->api_version != SELENE_RUNTIME_CURRENT_API_VERSION) {
        dlclose(r->library); free(r); return -1;
    }
    r->api = *p;
    int err = r->api.init_fn(&r->inner, n, start, argc - 2, argv + 2);
    if (err) { dlclose(r->library); free(r); return err; }
    runtime_instances++; *handle = r; return 0;
}
static SeleneErrno rt_exit(void *h) {
    Runtime *r = h;
    int err = r->api.exit_fn ? r->api.exit_fn(r->inner) : 0;
    dlclose(r->library); free(r); runtime_instances--; return err;
}
static SeleneErrno rt_start(void *h, uint64_t shot, uint64_t seed) {
    Runtime *r = h;
    phase = 0; pending_phase = -1; switches = 0; live = peak = 0;
    noisy_batches = ideal_batches = noisy_rzz = noisy_rxy = noisy_measure = 0;
    noisy_start = UINT64_MAX; noisy_end = 0;
    return r->api.shot_start_fn(r->inner, shot, seed);
}

/* A phase switches only after Narsil has emitted every pre-barrier batch and
 * Selene has delivered all its measurement results back to Narsil. */
typedef struct { RuntimeGetOperationHandle out; size_t count; } Forward;
#define FORWARD(name, decl, args) static void fw_##name decl { \
    Forward *f = h; f->count++; f->out.interface.name##_fn args; }
FORWARD(measure, (void *h,uint64_t q,uint64_t id), (f->out.instance,q,id))
FORWARD(measure_leaked, (void *h,uint64_t q,uint64_t id), (f->out.instance,q,id))
FORWARD(reset, (void *h,uint64_t q), (f->out.instance,q))
FORWARD(custom, (void *h,size_t tag,const void *p,size_t n), (f->out.instance,tag,p,n))
FORWARD(rzz, (void *h,uint64_t a,uint64_t b,double t), (f->out.instance,a,b,t))
FORWARD(rxy, (void *h,uint64_t q,double t,double p), (f->out.instance,q,t,p))
FORWARD(rz, (void *h,uint64_t q,double t), (f->out.instance,q,t))
FORWARD(rpp, (void *h,uint64_t a,uint64_t b,double t,double p), (f->out.instance,a,b,t,p))
static void fw_time(void *h,uint64_t t,uint64_t d) {
    Forward *f = h; f->out.interface.set_batch_time_fn(f->out.instance,t,d);
}
static SeleneErrno rt_next(void *h, RuntimeGetOperationHandle out) {
    Runtime *r = h;
    Forward f = {.out=out};
    RuntimeGetOperationHandle wrapped = {&f, {fw_measure,fw_measure_leaked,fw_reset,
        fw_custom,fw_time,fw_rzz,fw_rxy,fw_rz,fw_rpp}};
    int err = r->api.get_next_operations_fn(r->inner, wrapped);
    if (!err && !f.count && pending_phase >= 0) {
        phase = pending_phase; pending_phase = -1; switches++;
    }
    return err;
}
static SeleneErrno rt_barrier(void *h,const uint64_t *qs,uint64_t n,uint64_t sleep) {
    Runtime *r = h;
    if (sleep == BEGIN_NOISE || sleep == END_NOISE) {
        int want = sleep == BEGIN_NOISE;
        if (pending_phase >= 0 || switches != (want ? 0 : 1)) return -1;
        pending_phase = want;
        /* All surviving qubits at an experiment boundary are the data array.
         * Narsil implements local barriers, but not global_barrier. */
        return r->api.local_barrier_fn(r->inner, qs, n, 0);
    }
    return r->api.local_barrier_fn(r->inner, qs, n, sleep);
}
static SeleneErrno rt_alloc(void *h,uint64_t *q) {
    Runtime *r = h;
    if (live >= r->limit) {
        fprintf(stderr,"Sol program exceeds declared live-qubit limit %llu\n",(unsigned long long)r->limit);
        return -1;
    }
    int err = r->api.qalloc_fn(r->inner,q);
    if (!err) { live++; if (live > peak) peak = live; }
    return err;
}
static SeleneErrno rt_free(void *h,uint64_t q) {
    Runtime *r = h;
    if (!live) return -1;
    int err = r->api.qfree_fn(r->inner,q);
    if (!err) live--; return err;
}
#define DELEGATE(name, decl, args) static SeleneErrno rt_##name decl { \
    Runtime *r = h; return r->api.name##_fn args; }
DELEGATE(shot_end, (void *h), (r->inner))
DELEGATE(global_barrier, (void *h,uint64_t d), (r->inner,d))
DELEGATE(rxy_gate, (void *h,uint64_t q,double t,double p), (r->inner,q,t,p))
DELEGATE(rzz_gate, (void *h,uint64_t a,uint64_t b,double t), (r->inner,a,b,t))
DELEGATE(rz_gate, (void *h,uint64_t q,double t), (r->inner,q,t))
DELEGATE(rpp_gate, (void *h,uint64_t a,uint64_t b,double t,double p), (r->inner,a,b,t,p))
DELEGATE(measure, (void *h,uint64_t q,uint64_t *id), (r->inner,q,id))
DELEGATE(measure_leaked, (void *h,uint64_t q,uint64_t *id), (r->inner,q,id))
DELEGATE(reset, (void *h,uint64_t q), (r->inner,q))
DELEGATE(force_result, (void *h,uint64_t id), (r->inner,id))
DELEGATE(get_bool_result, (void *h,uint64_t id,int8_t *v), (r->inner,id,v))
DELEGATE(get_u64_result, (void *h,uint64_t id,uint64_t *v), (r->inner,id,v))
DELEGATE(set_bool_result, (void *h,uint64_t id,bool v), (r->inner,id,v))
DELEGATE(set_u64_result, (void *h,uint64_t id,uint64_t v), (r->inner,id,v))
DELEGATE(increment_future_refcount, (void *h,uint64_t id), (r->inner,id))
DELEGATE(decrement_future_refcount, (void *h,uint64_t id), (r->inner,id))
DELEGATE(custom_call, (void *h,uint64_t tag,const void *p,size_t n,uint64_t *v), (r->inner,tag,p,n,v))
DELEGATE(simulate_delay, (void *h,uint64_t d), (r->inner,d))
static SeleneErrno rt_metrics(void *h,uint8_t i,char *tag,uint8_t *type,uint64_t *value) {
    (void)h;
    const char *names[] = {"spiderwarp:peak_live_qubits","spiderwarp:phase_switches",
        "spiderwarp:noisy_batches","spiderwarp:ideal_batches","spiderwarp:noisy_rzz",
        "spiderwarp:noisy_rxy","spiderwarp:noisy_measurements","spiderwarp:protocol_duration_ns"};
    uint64_t values[] = {peak,(uint64_t)switches,noisy_batches,ideal_batches,noisy_rzz,
        noisy_rxy,noisy_measure,noisy_start == UINT64_MAX ? 0 : noisy_end - noisy_start};
    if (i >= sizeof(values)/sizeof(*values)) return 1;
    strcpy(tag,names[i]); *type = 2; *value = values[i]; return 0;
}
static const SeleneRuntimePluginDescriptorV1 runtime_descriptor = {
    sizeof(SeleneRuntimePluginDescriptorV1), SELENE_RUNTIME_CURRENT_API_VERSION,
    rt_init,rt_exit,rt_next,rt_start,rt_shot_end,rt_metrics,rt_alloc,rt_free,rt_barrier,
    rt_global_barrier,rt_rxy_gate,rt_rzz_gate,rt_rz_gate,rt_rpp_gate,rt_measure,
    rt_measure_leaked,rt_reset,rt_force_result,rt_get_bool_result,rt_get_u64_result,
    rt_set_bool_result,rt_set_u64_result,rt_increment_future_refcount,
    rt_decrement_future_refcount,rt_custom_call,rt_simulate_delay
};
const SeleneRuntimePluginDescriptorV1 *selene_runtime_get_plugin_descriptor_v1(void) {
    return &runtime_descriptor;
}

static SeleneErrno em_init(void **handle,uint64_t n,uint32_t argc,const char *const *argv) {
    if (model_instances || argc < 1) return -1;
    Model *m = calloc(1,sizeof(*m));
    if (!m) return -1;
    m->library = dlopen(argv[0],RTLD_NOW | RTLD_LOCAL);
    if (!m->library) { fprintf(stderr,"%s\n",dlerror()); free(m); return -1; }
    SeleneErrorModelPluginDescriptorV1 *p = descriptor(m->library,
        "selene_error_model_plugin_descriptor_v1","selene_error_model_get_plugin_descriptor_v1");
    if (!p || p->struct_size < sizeof(*p) || p->api_version != SELENE_ERROR_MODEL_CURRENT_API_VERSION) {
        dlclose(m->library); free(m); return -1;
    }
    m->api = *p;
    int err = m->api.init_fn(&m->inner,n,argc - 1,argv + 1);
    if (err) { dlclose(m->library); free(m); return err; }
    const char **ideal_args = calloc(argc - 1,sizeof(*ideal_args));
    if (!ideal_args) { m->api.exit_fn(m->inner); dlclose(m->library); free(m); return -1; }
    for(uint32_t i=1;i<argc;i++)
        ideal_args[i-1] = strncmp(argv[i],"--scale-overall=",16) == 0 ? "--scale-overall=0" : argv[i];
    err=m->api.init_fn(&m->ideal,n,argc - 1,ideal_args);
    free(ideal_args);
    if(err) { m->api.exit_fn(m->inner); dlclose(m->library); free(m); return err; }
    *handle = m; model_instances++; return 0;
}
static SeleneErrno em_exit(void *h) {
    Model *m = h; int err = m->api.exit_fn ? m->api.exit_fn(m->inner) : 0;
    if(m->api.exit_fn) { int other=m->api.exit_fn(m->ideal); if(!err) err=other; }
    dlclose(m->library); free(m); model_instances--; return err;
}
static SeleneErrno em_start(void *h,uint64_t shot,uint64_t seed) {
    Model *m = h; int err=m->api.shot_start_fn(m->inner,shot,seed);
    return err ? err : m->api.shot_start_fn(m->ideal,shot,seed);
}
static SeleneErrno em_end(void *h) {
    Model *m = h; int err=m->api.shot_end_fn(m->inner);
    return err ? err : m->api.shot_end_fn(m->ideal);
}

/* Sink used only to advance PECOS's elapsed-time bookkeeping in ideal regions. */
static SeleneErrno sink_one(void *h,uint64_t q) { (void)h;(void)q;return 0; }
static SeleneErrno sink_rz(void *h,uint64_t q,double t) { (void)t;return sink_one(h,q); }
static SeleneErrno sink_rxy(void *h,uint64_t q,double t,double p) { (void)p;return sink_rz(h,q,t); }
static SeleneErrno sink_rzz(void *h,uint64_t a,uint64_t b,double t) { (void)b;return sink_rz(h,a,t); }
static SeleneErrno sink_rpp(void *h,uint64_t a,uint64_t b,double t,double p) { (void)p;return sink_rzz(h,a,b,t); }
static void sink_bool(void *h,uint64_t id,bool v) { (void)h;(void)id;(void)v; }
static void sink_u64(void *h,uint64_t id,uint64_t v) { (void)h;(void)id;(void)v; }

static void ex_measure(void *h,uint64_t q,uint64_t id) {
    (void)h; (void)q; (void)id; noisy_measure++;
}
static void ex_leaked(void *h,uint64_t q,uint64_t id) {
    ex_measure(h,q,id);
}
static void ex_reset(void *h,uint64_t q) {
    (void)h; (void)q;
}
static void ex_custom(void *h,size_t tag,const void *p,size_t n) { (void)h;(void)tag;(void)p;(void)n; }
static void ex_time(void *h,uint64_t t,uint64_t d) {
    (void)h; if (t < noisy_start) noisy_start=t; if (t+d > noisy_end) noisy_end=t+d;
}
static void ex_rzz(void *h,uint64_t a,uint64_t b,double t) {
    (void)h; (void)a; (void)b; (void)t; noisy_rzz++;
}
static void ex_rxy(void *h,uint64_t q,double t,double p) {
    (void)h; (void)q; (void)t; (void)p; noisy_rxy++;
}
static void ex_rz(void *h,uint64_t q,double t) {
    (void)h; (void)q; (void)t;
}
static void ex_rpp(void *h,uint64_t a,uint64_t b,double t,double p) {
    (void)p; ex_rzz(h,a,b,t);
}
static SeleneErrno em_handle(void *h,RuntimeExtractOperationHandle batch,
                             SimulatorHandle sim,ErrorModelSetResultHandle results) {
    Model *m=h;
    SimulatorHandle sink={NULL,{.rxy_fn=sink_rxy,.rz_fn=sink_rz,.rzz_fn=sink_rzz,
        .rpp_fn=sink_rpp,.measure_fn=sink_one,.reset_fn=sink_one}};
    ErrorModelSetResultHandle discarded={NULL,{sink_bool,sink_u64}};
    if(phase) {
        noisy_batches++;
        RuntimeGetOperationHandle count={NULL,{ex_measure,ex_leaked,ex_reset,ex_custom,
            ex_time,ex_rzz,ex_rxy,ex_rz,ex_rpp}};
        batch.interface.extract_fn(batch.instance,count);
        int err=m->api.handle_operations_fn(m->ideal,batch,sink,discarded);
        if(err) return err;
        return m->api.handle_operations_fn(m->inner,batch,sim,results);
    }
    ideal_batches++;
    int err=m->api.handle_operations_fn(m->inner,batch,sink,discarded);
    if(err) return err;
    return m->api.handle_operations_fn(m->ideal,batch,sim,results);
}
static const SeleneErrorModelPluginDescriptorV1 model_descriptor={
    sizeof(SeleneErrorModelPluginDescriptorV1),SELENE_ERROR_MODEL_CURRENT_API_VERSION,
    em_init,em_exit,em_start,em_end,em_handle,NULL
};
const SeleneErrorModelPluginDescriptorV1 *selene_error_model_get_plugin_descriptor_v1(void) {
    return &model_descriptor;
}
