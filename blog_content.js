const blogContent = [
  {
    target: "server-boot-up",
    diagram_state: "init",
    title: "1. Server Launch",
    content: `
<p>A single-GPU SGLang server is composed of 3 processes: 1) HTTP server + Engine + TokenizerManager, 2) Scheduler 3) DetokenizerManager.<p>
<p>The TokenizerManager tokenizes the requests and sends them to the scheduler. The Scheduler receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager. The Detokenizer Manager detokenizes the output tokens and sends the result back to the Tokenizer Manager. <p>
`
  },
  {
    target: "_launch_subprocesses",
    diagram_state: "init",
    title: "1. Launch Subprocesses",
    content: `
<p>Because the Scheduler, Detokenizer, and Tokenizer run across separate processes, they can't communicate using standard variables in RAM. SGLang uses ZeroMQ (ZMQ) for ultra-fast, inter-process communication (IPC). <code>PortArgs</code> allocates a guaranteed set of unique, open ports (or file sockets paths) that all future subprocesses will use to send tensors and token dictionaries to one another.<p>
<p><p>
`
  },
  {
    target: "_launch_scheduler_processes",
    diagram_state: "init",
    title: "1. Launch Scheduler Subprocess",
    content: `
<p>It creates a single, one-way communication pipe (<code> reader, writer = mp.Pipe(duplex=False)</code>. It then creates the Scheduler process (<code>proc = mp.Process(..., writer,...)</code>) and gives it the writer to the pipe. When creating the Scheduler process, it calls <code>proc.start()</code> using <code> numa_utils.configure_subprocess(server_args, gpu_id)</code>, which is an SGLang utility function that does something very clever and hacky. Then it saves the process ID and reader in a list for use by the Engine. <p>
<p><p>
`
  },
  {
    target: "numa_utils.configure_subprocess",
    diagram_state: "init",
    title: "1. Bind Scheduler to Specific CPU",
    content: `
<p>This code is a specialized hack that forces Python's multiprocessing library to wrap child processes in the Linux numactl utility. Normally, when you start a child process in Python using the "spawn" method, Python looks up its own binary path (e.g., /usr/bin/python3, forcing the new process to be bound to a specific CPU socket before it can allocate any memory).<p>
<p>Python doesn't have a native  API to prefix the multiprocessing executable with an external shell command. This code solves that problem by tricking Python into running a temporary bash script instead of itself. First, it identifies which NUMA node the GPU we are using is connected to and constructs the flags that tell numactl to restrict all CPU threading and RAM allocations to the CPUs and RAM associated with a specific NUMA node. In my case, I ran <code>nvidia-smi topo -m</code>, got these results.... Thus, all 8 GPUs on my B300 are connected equally to all of the 240 CPUs so the Scheduler subprocess is allowed to run on all 240 CPU cores.<p>
`
  },
  {
    target: "_launch_subprocesses_part2",
    diagram_state: "init",
    title: "Launch Detokenizer Process",
    content: `
<p>_launch_subprocesses then launches the detokenizer subprocess and the subprocess watchdog. The subprocess watchdog was added much later in the development of the scheduler to fix an issue where the SGLang Scheduler subprocess would crash due to a C++ error but main HTTP server process continues running and cannot process inference requests. Read more about the scheduler subprocess watchdog here: https://github.com/sgl-project/sglang/issues/18421 <p>
`
  },

  {
    target: "_setup_and_run_http_server_function_call",
    diagram_state: "init",
    title: "Launch HTTP Server",
    content: `
<p>Then the HTTP server is launched in the main process<p>
`
  },
  {
    target: "_setup_and_run_http_server",
    diagram_state: "init",
    title: "Launch HTTP Server",
    content: `
<p>The HTTP server is run using Uvicorn, which is an ASGI (Asynchronous Server Gateway Interface).<p>
`
  },
  
  {
    target: "code-entry",
    diagram_state: "init",
    title: "1. The Server Event Loop",
    content: `
<p>In this post, I explain how the SGLang scheduler works in-depth and technically in-depth using a concrete example. I launch the SGLang server using this command:</p>

<pre><code class="language-bash">python -m sglang.launch_server \\
    --model-path Qwen/Qwen3-30B-A3B \\
    --moe-runner-backend triton \\
    --disable-cuda-graph \\
    --disable-piecewise-cuda-graph \\
    --port 30004</code></pre>

<p>The \`Scheduler\` class initialized in \`run_scheduler_process\` and it is given the \`server_args\`, \`port_args\`, \`gpu_id\`, \`tp_rank\`, \`moe_ep_rank\`, \`pp_rank\`, \`attn_cp_rank\`, \`moe_dp_rank\`, and \`dp_rank\`.</p>

The \`Scheduler\` sets all of its local variables to hold the important information needed to run the SGLang server.
`
  },
  {
    target: "init-scheduler",
    diagram_state: "queueing",
    title: "Initializing the scheduler",
    content: `
    <p>To understand the scheduler, there are six critical initializing function calls: <code>self.init_model_worker()</code>, <code>self.init_cache_with_memory_pool()</code>, <code>self.init_chunked_prefill()</code>, <code>self.init_schedule_policy()</code>, <code>self.init_overlap()</code>, and <code>self.init_request_dispatcher()</code>.</p>
    <p>Here is a comprehensive breakdown of how these functions set up the core infrastructure of the scheduler in SGLang:</p>
    
    <h3>1. <code>self.init_model_worker()</code></h3>
    <p>This function launches and configures the core model execution units (workers) and gathers configuration details required by the scheduler:</p>
    <ul>
      <li><strong>Worker Setup:</strong> It starts the primary Tensor Parallel (TP) worker and optionally initializes a draft worker if speculative decoding is enabled (<code>self.maybe_init_draft_worker()</code>).</li>
      <li><strong>Capacity Metrics:</strong> It requests constraints from the worker and sets hard boundaries on the scheduler's capacity, such as <code>max_running_requests</code>, <code>max_prefill_tokens</code>, and <code>max_total_num_tokens</code>.</li>
      <li><strong>Distribution Groups:</strong> It establishes all relevant orchestration and communication "process groups" for distributed inference, including Tensor Parallel (TP), DP Attention, Context Parallelism (CP), and Pipeline Parallelism (PP). These topology definitions enable the scheduler to correctly route requests.</li>
      <li><strong>Seed & Padding Setup:</strong> Initializes the function that pads input IDs correctly and ensures reproducible generation runs with a set seed.</li>
    </ul>
    
    <h3>2. <code>self.init_cache_with_memory_pool()</code></h3>
    <p>This function manages where and how tokens are securely mapped within GPU limits (KV caches):</p>
    <ul>
      <li><strong>Memory Pointers:</strong> It pulls the hybrid memory pool pointers from the worker <code>req_to_token_pool</code> (mapping requests to indices) and <code>token_to_kv_pool_allocator</code> (which points deeper to actual block availability).</li>
      <li><strong>Architecture-Specific Caching:</strong> It determines which caching structure the system should deploy based on the model properties. It selects between standard radices (<code>RadixCache</code>), experimental C++ variants (<code>RadixCacheCpp</code>), hierarchical implementations (<code>HiRadixCache</code> for storage offloading), or chunked caches if radices are disabled.</li>
      <li><strong>Disaggregation Needs:</strong> If the server is in specialized split stages ("Decode" mode) and allowed to offload caching, it initializes <code>DecodeKVCacheOffloadManager</code> here to pass KV states correctly.</li>
    </ul>
    
    <h3>3. <code>self.init_chunked_prefill()</code></h3>
    <p>This handles how the scheduler splits up long context inputs into manageable, scheduled segments to avoid out-of-memory (OOM) errors and head-of-line blocking:</p>
    <ul>
      <li><strong>Sizing Defaults:</strong> It calculates <code>self.chunked_prefill_size</code>. If chunking presents problems (e.g., Transformers backends simulating multimodal model inputs poorly), it dynamically turns itself off.</li>
      <li><strong>Dynamic Predictive Chunking:</strong> If Pipeline Parallelism (PP) is enabled along with <code>enable_dynamic_chunking</code>, it attempts to run a trace/profiler (<code>self.profile_and_init_predictor()</code>) to calculate real-time constraints for latency-bound dynamic chunking.</li>
    </ul>
    
    <h3>4. <code>self.init_schedule_policy()</code></h3>
    <p>This initializes the logic gatekeeper, which is responsible for determining who runs next and how much risk to take:</p>
    <ul>
      <li><strong>The Main Algorithm:</strong> It activates the primary <code>SchedulePolicy</code> class, handling queue sorting formats (FCFS, LPF, shortest-job-first, etc.) prioritizing cache access rates or structural priority weights.</li>
      <li><strong>The Prefill Delayer:</strong> If <code>enable_prefill_delayer</code> is set, it spins up a class dedicated to slowing or staggering prefill batches, halting dense prefills when the token bounds hit a low watermark footprint.</li>
      <li><strong>Token Ratios:</strong> Sets scaling factors like <code>init_new_token_ratio</code> and <code>new_token_ratio_decay</code>. These conservative ratios decay to smaller bounds mathematically during decoding steps to restrict token generation from greedily consuming available KV cache bounds during overlap events.</li>
    </ul>
    
    <h3>5. <code>self.init_overlap()</code></h3>
    <p>This function configures the asynchronous multi-stream mechanics required by Overlap Scheduling:</p>
    <ul>
      <li><strong>CUDA Context Hooks:</strong> It maps CUDA device streams logic into Python variables to guarantee overlap is executed cleanly. It creates the Python wrapper <code>forward_stream_ctx</code> and establishes a secondary independent stream: <code>copy_stream</code>.</li>
      <li><strong>The Future Map:</strong> The core overlap enabler. If overlap is enabled, it configures <code>self.future_map</code>. The <code>FutureMap</code> coordinates tracking PyTorch asynchronous events in flight, matching requested sizes and sequences logically inside the Python event loop without pausing generation for device synchronizations.</li>
      <li><strong>Ping-Pong Buffers:</strong> Initializes double-buffers (<code>batch_record_buf</code> of length 2) that efficiently track states flipping between previous runs and subsequent overlaps.</li>
    </ul>
    
    <h3>6. <code>self.init_request_dispatcher()</code></h3>
    <p>This acts as the translation layer separating ZMQ network inputs from inner code execution mechanisms:</p>
    <ul>
      <li><strong>Type Switcher:</strong> It registers a massive map via <code>TypeBasedDispatcher</code>. Based on the distinct programmatic type of an incoming object, it reroutes the variable directly to the appropriate method (e.g., passing a <code>TokenizedGenerateReqInput</code> object instantly to <code>handle_generate_request()</code>).</li>
      <li><strong>Control Calls:</strong> Aside from generation inputs, this ties up utility endpoints. This includes <code>AbortReq</code>, distributed weights injections (<code>InitWeightsUpdateGroupReqInput</code>, <code>UpdateWeightsFromTensorReqInput</code>), RPC calls (<code>RpcReqInput</code>), or diagnostic pause instructions (<code>PauseGenerationReqInput</code>), effectively serving as the scheduler's internal API surface.</li>
    </ul>
    `
  },
  {
    target: "send-init-info-to-parent-process",
    diagram_state: "queueing",
    title: "3. The Step Execution",
    content: `
<p>The core heartbeat of the scheduler is the \`step()\` function. Called on every iteration of the loop, it manages the full lifecycle of a forward pass.</p>
<p>It explicitly orchestrates memory management, request batching, and finally delegates execution to the GPU workers.</p>
`
  },
  {
    target: "event-loop",
    diagram_state: "caching",
    title: "4. Memory Management",
    content: `
<p>Before batching, the scheduler jumps to checking its Radix Cache to prevent Out-Of-Memory errors.</p>
<p>If the VRAM cache usage exceeds a safe threshold (like 95%), it proactively offloads the least recently used KV-cache blocks to the CPU, freeing up space for the active generation step.</p>
`
  },
  {
    target: "dispatch_event_loop",
    diagram_state: "forward",
    title: "5. Batching Requests",
    content: `
<p>Finally, it looks back at the waiting queue. Knowing exactly how much memory is available, it continuously pops requests to form an optimal batch.</p>
<p>The newly formed batch is then handed over to the GPU for the forward pass, completing the cycle.</p>
`
  }
,
  {
    target: "event_loop_normal",
    diagram_state: "init",
    title: "event_loop_normal",
    content: `
<p>TODO</p>
`
  },
  {
    target: "recv_requests",
    diagram_state: "init",
    title: "recv_requests",
    content: `
<p>This <code>recv_requests</code> function is a critical piece of the <code>Scheduler</code> that orchestrates distributed inference. Its primary job is to fetch new incoming requests and synchronize them across multiple GPUs or nodes when using Pipeline Parallelism (PP), Tensor Parallelism (TP), or Data Parallelism (DP). In our single GPU setup, it does  not do very much. Since we are using a generation model, <code>recv_requests</code> will return a list of <code>TokenizedGeneratedReqInput</code> objects.</p>
<p>This stays exactly the same. The scheduler still needs to decide if it's a good time to accept new requests or if it should focus purely on generation (overlap scheduling). If the recv_skipper is None, then it always accepts new requests. By default, recv_skipper is None for all SGLang servers. However, it can be set during server startup by setting the flag --scheduler-recv-interval greater than 1. Setting it greater than 1 will decrease the rate that the SGLang server polls for ZMQ requests for other components of the SGLang system, making it run faster. The <code>handle</code> function determines whether we should not poll for requests on this iteration of the scheduler loop: by default 1000 decode steps are allowed for every 1 prefill step. This threshold and proportion can be controlled via the environment variables <code>SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT</code>, <code>SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE</code>, <code>SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY</code>, and <code>scheduler_recv_interval</code> flag If the skipper says "stop," it returns an empty list <code>[]</code>. <p>
<p>Because <code>pp_rank == 0</code>, <code>attn_tp_rank == 0</code>, and <code>attn_cp_rank == 0</code> are all True, the execution enters the very first </code>if<code> block.<p>
<p>Since this is the only GPU, it acts as the "master." It rapidly polls the active ZMQ sockets (recv_from_tokenizer and recv_from_rpc) in a non-blocking way. It grabs any new requests waiting in the operating system's network buffers and appends them to local memory. The else blocks containing point-to-point communication (point_to_point_pyobj) for later pipeline stages are completely ignored.<p>
<p>Both loops act like vacuum cleaners: they run continuously to suck up all pending requests, but are forced to shut off either when the "bag is full" or when "there is nothing left to vacuum". Concretely, the SGLang Scheduler defines an environment variable SGLANG_SCHEDULER_MAX_RECV_PER_POLL with default value set to infinity. The While loop checks if this current call to recv_requests has formed a list of generation requests (TokenizedGenerateReqInput) that exceeds the limit per call to recv_requests. If it has, then we break out of this infinite polling loop that is grabbing requests and adding them our list of requests to process. However, since there is no limit in the standard case, it seems like if there are enough requests flooding the server, it could get stuck in an infinite polling loop. This is virtually impossible though because the while loop iterates much faster than the fronend feeds requests into the ZeroMQ buffer.<p>
<p>There are two loops because there are two request sources that must be taken from: TokenizerManager and Engine. The TokenizerManager has 21 different request types that it sends to the Scheduler loop: 1) _send_one_request, _send_batch_request, abort_request, router_worker_obj, close_session (CloseSessionReqInput), pause_generation (PauseGenerationReqInput), continue_generation (ContinueGenerationReqInput),  input_blocker_guard_region (BlockReqInput(BlockReqType.BLOCK)), input_blocker_guard_region (BlockReqInput(BlockReqType.UNBLOCK)), freeze_gc (FreezeGCReq), open_session (OpenSessionReqInput), _wait_for_model_update_from_disk (UpdateWeightFromDiskReqInput), _handle_batch_output (WatchLoadUpdateReq), update_active_ranks (ActiveRanksOutput)   <p>
<p>The recv_reqs function then returns the list of mostly TokenizedGenerateReqInput objects, but technically all request objects that we listed above. <p>
`
  },
  {
    target: "process_input_requests",
    diagram_state: "init",
    title: "process_input_requests",
    content: `
<p>Immediately after the scheduler pulls a batch of new incoming requests from the ZeroMQ pipes via <code>recv_requests()</code>, it passes them to this function.

Its primary purpose is to ingest, route, and perform initial processing for all new requests before they are added to the waiting queues or acted upon by the active inference step. It iterates over the list of <code>recv_reqs</code> and dispatches the request via <code>_request_dispatcher</code> which uses a standard Python dictionary to map the request to its handler and then call its handler. For iterations where the request is a generation request (<code>TokenizedGenerateReqInput</code> or <code>BatchTokenizedGenerateReqInput</code>,  <code>_request_dispatcher</code> will return <code> None </code> so no more work will be done on that iteration.</p>
`
  },
  {
    target: "_request_dispatcher",
    diagram_state: "init",
    title: "_request_dispatcher",
    content: `
<p>Let's delve deeper into what happens when a request is dispatched (<code>output = self._request_dispatcher(recv_req) </code>). The request type is matched against one of 41 request types. For a generation request(<code>TokenizedGenerateReqInput</code> or <code>BatchTokenizedGenerateReqInput</code>),  <code>handle_generate_request()</code> is called.  </p>
`
  },
  {
    target: "handle_generate_request",
    diagram_state: "init",
    title: "Handle Generate Request",
    content: `
<p>The <code>handle_generate_request()</code> function is responsible for converting a raw incoming generate request (a simple data class/dictionary from the API server) into a rich, stateful Req object that the SGLang engine can schedule and execute. SGLang allows "sessions," which let users maintain conversational state across multiple requests without resending the entire context. The function first checks if a session_id is present.</p>
<p>If there is no session ID, it immediately forms a new <code> Req </code> instance with ALL of the information for this specific user's generation request. If there is a session ID (in the request), then a request is created from the session. If there is a session ID, but the corresponding session is not found, the scheduler creates a barebones <code> Req </code> to use to abandon the request. (what exactly does it return to the user?? and what does it bake into the request to make it get abandoned??) <p>
<p> <code> self.init_req_max_new_tokens(req)</code> checks how many tokens the user wants to generate (<code> max_new_tokens </code>) and clamps it so the total prompt length + generated tokens doesn't exceed the model's max sequence length limits.<p> 
`
  },
  {
    target: "init_req_max_new_tokens",
    diagram_state: "init",
    title: "Handle Generate Request",
    content: `
<p>Specifically, <code> init_req_max_new_tokens </code> sets the <code>max_new_tokens</code> to the smaller of either: the context size of that model (derived from the model config file) or the requested <code>max_tokens</code><p>
`
  },
  {
    target: "self._add_request_to_queue(req)",
    diagram_state: "init",
    title: "Handle Generate Request",
    content: `
<p>This function call acts as a crucial routing mechanism for new requests entering the inference engine. Its primary job is to perform final validation and preparation on a <code>Req</code> object before placing it into the appropriate queue based on the instance's Disaggregation Mode (standalone server, prefill-only node, decode-only node)<p>
<p>The function does two things: initiates fetching tensors from the KV cache and append the <code> Req </code> object to the Scheduler's <code> waiting_queue </code>. (which one of these function calls is the path that we actually take?? How many different queues do we have??)<p>
`
  },
  {
    target: "self._prefetch_kvcache(req)",
    diagram_state: "init",
    title: "Handle Generate Request",
    content: `
<p>This function is solely dedicated to Hicache<p>

`
  },
  {
    target: "cancel_bubble_timer",
    diagram_state: "init",
    title: "cancel_bubble_timer",
    content: `
<p>If the Engine is paused, the event loop pauses the timer that measures GPU idle time ("bubbles") via <code>cancel_bubble_timer()</code> (which PR introduced the ability to pause the Engine??</p>
`
  },
  {
    target: "get_next_batch_to_run",
    diagram_state: "init",
    title: "get_next_batch_to_run",
    content: `
<p>The function's primary goal is to return a ScheduleBatch representing the next unit of work for the GPU, or None if the scheduler is idle. It achieves this by cleaning up expired or finished requests, merging the outcomes of previous executions into the persistent running_batch, and deciding whether to initiate a new Prefill batch (prioritized) or to continue Decoding for the existing running_batch. </p>
<p>The first actions it performs are health checks to prune requests that have stalled. It explicitly sweeps and aborts requests stuck in waiting_queue and the running_batch that have exceeded predefined maximum thresholds (e.g., waiting too long or running for an extremely high number of iterations). <p>
`
  },
  {
    target: "_abort_on_waiting_timeout",
    diagram_state: "init",
    title: "_abort_on_waiting_timeout",
    content: `
<p>This function manages the Queue Eviction phase. It iterates through the <code>Req </code> objects in the waiting_queue (requests that have been received by the engine but have not yet received hardware allocation for their first chunk of prefilling) and removes those that have been waiting longer than the environment <code> SGLANG_REQ_WAITING_TIMEOUT</code>. By default, there is no timeout for requests. To set a timeout, <code> SGLANG_REQ_WAITING_TIMEOUT</code> must be set to greater than 0.  This function iterates through the entire <code> waiting_queue </code> twice: once to collected expired requests and again to form the new filtered <code> waiting_queue </code>. This may be an inefficiency that should be addressed. (see if it was addressed??)</p>
`
  },
  {
    target: "_abort_on_running_timeout",
    diagram_state: "init",
    title: "_abort_on_running_timeout",
    content: `
<p>This function manages the Execution Eviction phase. It targets requests actively loaded into VRAM inside the running_batch. These are requests that passed the waiting phase, acquired GPU caches, and have begun (or mostly finished) prefilling / generating tokens, but are taking too long to completely finish their text generation.</p>
<p>This function iterates through all the requests (<code> Req </code> instances) in <code> self.running_batch.reqs </code> and sets <code> req.to_finish = FINISH_ABORT(..) </code> if it has exceeded the time limit set by the environment variable <code> SGLANG_REQ_RUNNING_TIMEOUT </code><p>
<p> Marking the <code> Req </code> with <code> FINISH_ABORT </code> has downstream effects. <code> handle_generate_request </code> explicitly checks if a <code> Req </code> has finish status <code> FINISH_ABORT </code>, but instead of catching the error and trying to manually construct an <code> HTTP 400 Bad Request </code> network response and streaming it over the ZeroMQ pipe, <code> handle_generate_request </code> adds it to <code> waiting_queue </code> like it is a healthy <code> Req </code>. <p>
<p>On the same iteration of the event loop, this request marked as <code> FINISH_ABORT </code> is dealt with in the <code> process_batch_result(batch,result) </code> (come back to this??) <p>
`
  },
  {
    target: "get_next_batch_to_run_part2",
    diagram_state: "init",
    title: "_abort_on_running_timeout",
    content: `
<p>This specific part of the get_next_batch_to_run scheduling logic plays a critical role in managing Chunked Prefill and separating requests that are still digesting their prompt from requests that are ready to generate text (Decode).<p>
<p>There are two important variables to consider here: <code> self.running_batch</code> and <code> self.last_batch </code>, which are both ScheduleBatch instances. The underlying intention of this block is to prepare <code> self.last_batch </code> (which just finished a forward pass) so it can be merged into the <code> self.running_batch</code> <p>
<p><code> chunked_req_to_exclude = set() </code> initializes a set of <code> Req </code> instances that must not be allowed to transition into the <code> running_batch </code> during the upcoming merge process. <p>
<p> When a single request features a massive prompt (e.g., 60,000 tokens), SGLang breaks that prompt down into smaller "chunks" (e.g., 4096 tokens per run). During this time, the active request is stored inside self.chunked_req. If self.chunked_req is not None, it means the engine is currently in the middle of processing a large prompt across multiple forward passes.<p>
<p>chunked_req_to_exclude.add(self.chunked_req): Because this request has only finished a fraction of its prompt so far, it is fundamentally unprepared to start generating new tokens. By adding it to the exclude set, the execution ensures that further down in the get_next_batch_to_run logic, when self.last_batch.filter_batch(chunked_req_to_exclude) is called, this chunked request is forcefully stripped out before the remainder of last_batch merges into running_batch. <p>
<p>This single line <code>self.stash_chunked_request(self.chunked_req)</code> does incredibly important bookkeeping for the Radix Attention (KV Cache) memory manager.<p>
`
  },
  {
    target: "self.stash_chunked_request(self.chunked_req)",
    diagram_state: "init",
    title: "_abort_on_running_timeout",
    content: `
<p>When a large request finishes processing round 1 of its chunks, the KV caching engine needs to know that the tokens it just calculated aren't the end of the prompt, but it also needs to safely store them so round 2 can use them immediately on the next inference cycle.

Calling cache_unfinished_req guarantees that the computed prefix tokens from the latest chunk are explicitly registered into the Radix Tree cache as an intermediate state. This does two main things:

Safety: It prevents the eviction mechanisms from accidentally freeing the KV cache of this intermediate state while the scheduler works on other things between the chunks.
Setup: It allows the next forward pass for chunked_req to cleanly resume from the exact edge of the tree cache node without having to reload or recompute the state.<p>

<p>If we navigate to <code>cache_unfinished_req </code>...<p>
`
  },
  {
    target: "self.stash_chunked_request(self.chunked_req)",
    diagram_state: "init",
    title: "_abort_on_running_timeout",
    content: `
<p>Before looking at the operation itself, it is important to understand how SGLang maps your Req object to the actual physical GPU memory storing the KV cache allocations.

self.req_to_token_pool: This is the global memory management object that orchestrates physical token allocations for all active requests.
self.req_to_token_pool.req_to_token: This is a massive 2D PyTorch Tensor (acting as a mapping table) with the shape (max_concurrent_requests, max_sequence_length). When you look up a specific row, you get back the exact physical GPU memory pool index locations (the token_pool indices) where the KV cache matrices for that sequence are saved.
req.req_pool_idx: This is the unique row ID integer assigned to this specific Req within the mapping table above.<p>
<p>req.fill_ids is a Python list storing all the token IDs that the request has processed so far (a combination of the original prompt input + any generated output).<p>
<p>What this slicing operation does: It goes to the req_to_token mapping table, grabs the row representing this request (req.req_pool_idx), and slices the row from index 0 to exactly len(req.fill_ids).<p>
<p>kv_indices is now a 1D tensor containing the exact physical KV cache memory pointers for the intermediate prompt chunk that just evaluated. (give a concrete example??<p>
<p>This function operates purely at the bookkeeping level. It literally just grabs the numerical indices (the "pointers") and copies them over to req.prefix_indices. It does not alter, move, compute, or mutate the actual Key and Value embedding tensors themselves. Now, this specific <code> Req </code> knows exactly where in the server-wide KV cache its KV tensors for this chunk of the prompt are stored<p>
<p> (figure out where these pointers are used??)<p>
`
  },
  {
    target: "get_next_batch_to_run_part2",
    diagram_state: "init",
    title: "stash_chunked_request",
    content: `
<p>The primary job of this code block is to decide whether the engine should perform a prefill pass or a decode pass on the next iteration, while carefully managing request states when transitioning between the two.</p>
`
  },
  {
    target: "get_next_batch_to_run_part2.1",
    diagram_state: "init",
    title: "stash_chunked_request",
    content: `
<p>This block executes if the engine just finished processing a prefill batch (forward_mode.is_extend()) and the hisparse feature is disabled (since hisparse has a custom integration handled just above this code). </p>
<p>The goal here is to take the requests that just had their context prefilled and add them to the running_batch (which holds all requests actively undergoing token decoding).<p>
<p>Before blindly merging, it needs to exclude requests that shouldn't transition to the standard decode phase. For example, if a request was part of a chunked prefill but hasn't finished evaluating all of its chunks, or if the request is being handled by decentralized LLM (DLLM) worker logic.<p>
<p>It calls filter_batch() on the newly completed prefill batch to remove any requests that are finished (e.g., they hit an abort or reached max length right after prefill) and removes the excluded chunked requests. If any requests were dropped (the new size is less than last_bs), it flags that self.running_batch is no longer completely full, implicitly signaling that the scheduler has room to accept new requests soon. <p>
<p>(how is the ideal batch size determined?? does it change over time??) (what is the filter batch logic??)<p>
<p>Finally, it merges the clean last_batch into the running_batch. If running_batch already has decoding requests, it fuses them together. This marks the transition point where prefilled requests are officially promoted to the decode phase.<p>
`
  },
  {
    target: "self.last_batch.filter_batch",
    diagram_state: "init",
    title: "self.last_batch.filter_batch",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.running_batch.merge_batch(self.last_batch)",
    diagram_state: "init",
    title: "self.running_batch.merge_batch(self.last_batch)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "get_next_batch_to_run_part2.2",
    diagram_state: "init",
    title: "stash_chunked_request",
    content: `
<p>It attempts to pull waiting requests from the <code> waiting_queue </code> (??) to construct a brand new prefill batch. <p>
`
  },
  {
    target: "new_batch = self.get_new_batch_prefill()",
    diagram_state: "init",
    title: "new_batch = self.get_new_batch_prefill()",
    content: `
<p>The overarching goal of this function is to determine if a new batch of prefill requests should be formed and scheduled, while potentially intentionally delaying these prefills to optimize overall system throughput and decode latencies. </p>
<p>This function will either return <code> None </code> or a new <code< ScheduleBatch </code> object. If it returns <code> None </code>, then this means one of four things: 1) it is more optimal to run decode right now 2) There are literally no prefill requests right now 3) If <code> running_batch </code> already contains the absolute maximum number of concurrent requests allowed (<code> max_running_requests </code>) 4) If there isn't enough memory in the KV Cache token pools to safely allocate blocks for a new prompt, it is better to finish decoding some prompts and evict their KV caches once they are done (is that really how eviction works??)<p>
<p>The Prefill Delayer is an advanced scheduling optimization in SGLang designed to solve the "prefill stalls decode" problem.<p>
<p>In standard continuous batching, evaluating a new prompt (prefilling) is highly compute-intensive, whereas predicting the next token (decoding) for existing requests is highly memory-bandwidth constrained. If a large prefill comes into the queue while the GPU is happily decoding tokens for active requests, simply adding that large prefill into the batch will "hijack" the GPU's compute. This stalls the active decoding requests, causing massive latency spikes (jitter) for users currently receiving streaming text. The Prefill Delayer intentionally "delays" new prefills for a set number of forward passes (max_delay_passes), giving priority to active decode requests so they can output tokens smoothly without interruption.<p>
<p><p>
`
  },
  {
    target: "new_batch is not None",
    diagram_state: "init",
    title: "new_batch is not None",
    content: `
<p>TODO</p>
`
  },
  {
    target: "new_batch is None",
    diagram_state: "init",
    title: "new_batch is None",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.running_batch is not empty and not self.running_batch.is_prefill_only",
    diagram_state: "init",
    title: "self.running_batch is not empty and not self.running_batch.is_prefill_only",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.running_batch = self.update_running_batch(self.running_batch)",
    diagram_state: "init",
    title: "self.running_batch = self.update_running_batch(self.running_batch)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "run_batch",
    diagram_state: "init",
    title: "run_batch",
    content: `
<p>TODO</p>
`
  },
  {
    target: "run_batch no overlap branch",
    diagram_state: "init",
    title: "run_batch no overlap branch",
    content: `
<p>TODO</p>
`
  },
  {
    target: "no overlap forward batch generation",
    diagram_state: "init",
    title: "no overlap forward batch generation",
    content: `
<p>TODO</p>
`
  },
  {
    target: "no overlap update_cache_from_scheduler",
    diagram_state: "init",
    title: "no overlap update_cache_from_scheduler",
    content: `
<p>TODO</p>
`
  },
  {
    target: "process_batch_result",
    diagram_state: "init",
    title: "process_batch_result",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part1",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part1",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part2",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part2",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part3",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part3",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part4",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part4",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part5",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part5",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part6",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part6",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_prefill(batch, result)_part7",
    diagram_state: "init",
    title: "self.process_batch_result_prefill(batch, result)_part7",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)_part1",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)_part1",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)_part2",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)_part2",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)_part3",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)_part3",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)_part4",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)_part4",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.process_batch_result_decode(batch, result)_part5",
    diagram_state: "init",
    title: "self.process_batch_result_decode(batch, result)_part5",
    content: `
<p>TODO</p>
`
  },
  {
    target: "on_idle",
    diagram_state: "init",
    title: "on_idle",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self_check_during_busy",
    diagram_state: "init",
    title: "self_check_during_busy",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.record_batch_in_overlap(model_worker_batch)",
    diagram_state: "init",
    title: "self.record_batch_in_overlap(model_worker_batch)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "model_worker_batch.sampling_info.copy_for_forward()",
    diagram_state: "init",
    title: "model_worker_batch.sampling_info.copy_for_forward()",
    content: `
<p>TODO</p>
`
  },
  {
    target: "future_indices = self.future_map.alloc_future_indices(bs)",
    diagram_state: "init",
    title: "future_indices = self.future_map.alloc_future_indices(bs)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "forward_batch_generation",
    diagram_state: "init",
    title: "forward_batch_generation",
    content: `
<p>TODO</p>
`
  },
  {
    target: "self.future_map.store_to_map(future_indices, batch_result)",
    diagram_state: "init",
    title: "self.future_map.store_to_map(future_indices, batch_result)",
    content: `
<p>TODO</p>
`
  },
  {
    target: "batch_result.copy_to_cpu(return_logprob=batch.return_logprob)",
    diagram_state: "init",
    title: "batch_result.copy_to_cpu(return_logprob=batch.return_logprob)",
    content: `
<p>TODO</p>
`
  }
];
