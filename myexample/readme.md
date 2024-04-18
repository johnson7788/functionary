Functionary 是一种可以解释和执行函数/插件的语言模型。
该模型确定何时执行函数，无论是并行还是串行，并且可以理解它们的输出。它仅根据需要触发功能。函数定义以 JSON 架构对象的形式给出，类似于 OpenAI GPT 函数调用。

# message中的角色必须是system,user, assistant,role这几种

# 运行
vllm运行模型，需要把模型放到目录下
ls meetkai/functionary-small-v2.4 
占用显存： 21672MiB
python3 server_vllm.py --model "meetkai/functionary-small-v2.4" --host 0.0.0.0 --max-model-len 8192

可能少些占用显存
python3 server_vllm.py --model "meetkai/functionary-small-v2.4" --host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.95

python3 server.py --load_in_8bit True --model "meetkai/functionary-small-v2.4" --device cuda:0

meetkai/functionary-small-v2.4 : 使用的是MistralForCausalLM

2.2是用的llama7B

# 测试
python myexample/myfunction.py
输出结果
```python
display_id='c9550697-3743-44e0-be45-66b437b22f10' content='' finished=True has_displayed=False
None
调用了函数：翻转硬币, 翻转的结果是: tails
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content='' finished=False has_displayed=False
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content='' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin landed' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin landed on' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin landed on t' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin landed on tails' finished=False has_displayed=True
display_id='566097a8-8ad3-4508-99ee-0a609fbbd517' content=' The coin landed on tails.' finished=False has_displayed=True
None
```

## 测试中文
display_id='a9e0ed4c-1baf-4966-b4af-3c1133b01ebc' content='' finished=True has_displayed=False
None
调用了函数：翻转硬币, 翻转的结果是: tails
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content='' finished=False has_displayed=False
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content='' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' ' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结果' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结果是' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结果是尾' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结果是尾部' finished=False has_displayed=True
display_id='5a2b3db5-4ee8-4893-b894-b4ba9e006b84' content=' 翻转硬币的结果是尾部。' finished=False has_displayed=True
None

# 使用llama cpp的方式运行
python myexample/llama_cpp.py
安装的依赖包没有使用CUDA pip install llama-cpp-python

输出结果:
```python
ssh://johnson@l0:22/home/wac/johnson/anaconda3/envs/qlora/bin/python -u /home/wac/johnson/.pycharm_helpers/pydev/pydevd.py --multiproc --qt-support=auto --client 0.0.0.0 --port 34023 --file /media/wac/backup/john/johnson/functionary/myexample/my_llama_cpp.py
/home/wac/johnson/.pycharm_helpers/pydev/pydevd.py:1844: DeprecationWarning: currentThread() is deprecated, use current_thread() instead
  dummy_thread = threading.currentThread()
Connected to pydev debugger (build 211.7442.45)
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.f16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = .
llama_model_loader: - kv   2:                           llama.vocab_size u32              = 32004
llama_model_loader: - kv   3:                       llama.context_length u32              = 32768
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                          llama.block_count u32              = 32
llama_model_loader: - kv   6:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   7:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   9:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  10:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  11:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  12:                          general.file_type u32              = 1
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  14:                      tokenizer.ggml.tokens arr[str,32004]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  15:                      tokenizer.ggml.scores arr[f32,32004]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,32004]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  19:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {% for message in messages %}\n{% if m...
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type  f16:  226 tensors
llm_load_vocab: special tokens definition check successful ( 263/32004 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32004
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 13.49 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = .
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 2 '</s>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MiB
llm_load_tensors:        CPU buffer size = 13813.08 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   512.00 MiB
llama_new_context_with_model: KV self size  =  512.00 MiB, K (f16):  256.00 MiB, V (f16):  256.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute buffer size =   296.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 1
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | 
Model metadata: {'tokenizer.chat_template': '{% for message in messages %}\n{% if message[\'role\'] == \'user\' or message[\'role\'] == \'system\' %}\n{{ \'<|from|>\' + message[\'role\'] + \'\n<|recipient|>all\n<|content|>\' + message[\'content\'] + \'\n\' }}{% elif message[\'role\'] == \'tool\' %}\n{{ \'<|from|>\' + message[\'name\'] + \'\n<|recipient|>all\n<|content|>\' + message[\'content\'] + \'\n\' }}{% else %}\n{% set contain_content=\'no\'%}\n{% if message[\'content\'] is not none %}\n{{ \'<|from|>assistant\n<|recipient|>all\n<|content|>\' + message[\'content\'] }}{% set contain_content=\'yes\'%}\n{% endif %}\n{% if \'tool_calls\' in message and message[\'tool_calls\'] is not none %}\n{% for tool_call in message[\'tool_calls\'] %}\n{% set prompt=\'<|from|>assistant\n<|recipient|>\' + tool_call[\'function\'][\'name\'] + \'\n<|content|>\' + tool_call[\'function\'][\'arguments\'] %}\n{% if loop.index == 1 and contain_content == "no" %}\n{{ prompt }}{% else %}\n{{ \'\n\' + prompt}}{% endif %}\n{% endfor %}\n{% endif %}\n{{ \'<|stop|>\n\' }}{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}{{ \'<|from|>assistant\n<|recipient|>\' }}{% endif %}', 'tokenizer.ggml.add_eos_token': 'false', 'tokenizer.ggml.padding_token_id': '2', 'tokenizer.ggml.unknown_token_id': '0', 'tokenizer.ggml.eos_token_id': '2', 'tokenizer.ggml.model': 'llama', 'general.architecture': 'llama', 'llama.rope.freq_base': '1000000.000000', 'llama.context_length': '32768', 'general.name': '.', 'llama.vocab_size': '32004', 'general.file_type': '1', 'tokenizer.ggml.add_bos_token': 'true', 'llama.embedding_length': '4096', 'llama.feed_forward_length': '14336', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.rope.dimension_count': '128', 'tokenizer.ggml.bos_token_id': '1', 'llama.attention.head_count': '32', 'llama.block_count': '32', 'llama.attention.head_count_kv': '8'}
Using gguf chat template: {% for message in messages %}
{% if message['role'] == 'user' or message['role'] == 'system' %}
{{ '<|from|>' + message['role'] + '
<|recipient|>all
<|content|>' + message['content'] + '
' }}{% elif message['role'] == 'tool' %}
{{ '<|from|>' + message['name'] + '
<|recipient|>all
<|content|>' + message['content'] + '
' }}{% else %}
{% set contain_content='no'%}
{% if message['content'] is not none %}
{{ '<|from|>assistant
<|recipient|>all
<|content|>' + message['content'] }}{% set contain_content='yes'%}
{% endif %}
{% if 'tool_calls' in message and message['tool_calls'] is not none %}
{% for tool_call in message['tool_calls'] %}
{% set prompt='<|from|>assistant
<|recipient|>' + tool_call['function']['name'] + '
<|content|>' + tool_call['function']['arguments'] %}
{% if loop.index == 1 and contain_content == "no" %}
{{ prompt }}{% else %}
{{ '
' + prompt}}{% endif %}
{% endfor %}
{% endif %}
{{ '<|stop|>
' }}{% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|from|>assistant
<|recipient|>' }}{% endif %}
Using chat eos_token: </s>
Using chat bos_token: <s>
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.



User: what's the weather like in Santa Cruz, CA compared to Seattle, WA?
/home/wac/johnson/anaconda3/envs/qlora/lib/python3.10/site-packages/pydantic/json_schema.py:2099: PydanticJsonSchemaWarning: Default value annotation=NoneType required=True description='The city and state, e.g., San Francisco, CA' is not JSON serializable; excluding default from JSON schema [non-serializable-default]
  warnings.warn(message, PydanticJsonSchemaWarning)
Tools: 
[
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "parameters": {
        "properties": {
          "location": {
            "type": "string"
          }
        },
        "type": "object",
        "required": []
      },
      "description": "Get the current weather"
    }
  }
]

  𝑓  get_current_weather({"location": "Santa Cruz, CA"})  ->  {'temperature': 74, 'units': 'F', 'weather': 'windy'}
  𝑓  get_current_weather({"location": "Seattle, WA"})  ->  {'temperature': 70, 'units': 'F', 'weather': 'sunny'}

Llama.generate: prefix-match hit
Assistant: The current weather in Santa Cruz, CA is windy with a temperature of 74°F. In Seattle, WA, the weather is sunny with a temperature of 70°F.

Process finished with exit code 0

```

安装CUDA的llama-cpp-python：（没成功)
pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123 -U --force-reinstall


# 数据集格式
https://github.com/MeetKai/functionary/blob/main/tests/test_case_v2.json