## 快速使用：


```shell                                                                                                                                                                            
# 1. 设置必要变量                                                                                                                                                               
export MEGATRON_PATH=/path/to/Megatron-LM                                                                                                                                       
export CKPT_PATH=/path/to/checkpoint                                                                                                                                            
export TOKENIZER_MODEL=/path/to/tokenizer.model                                                                                                                                 
                                                                                                                                                                                
# 2. 选择模式运行                                                                                                                                                               
bash npu_megatron_llm_eval.sh single   # 单卡                                                                                                                                   
bash npu_megatron_llm_eval.sh dp       # 4卡数据并行                                                                                                                            
bash npu_megatron_llm_eval.sh tp       # 2卡张量并行                                                                                                                            
bash npu_megatron_llm_eval.sh ep       # 4卡专家并行(MoE)                                                                                                                       
                                                                                                                                                                                
# 3. 自定义参数                                                                                                                                                                 
NUM_DEVICES=8 TASKS="hellaswag,arc_easy" bash npu_megatron_llm_eval.sh dp                                                                                                       
TP_SIZE=4 NUM_DEVICES=4 bash npu_megatron_llm_eval.sh tp                                                                                                                        
EXTRA_ARGS="--no-rope-fusion --trust-remote-code" bash npu_megatron_llm_eval.sh custom                                                                                          
```

## 脚本特性：

- 自动设置 ASCEND_RT_VISIBLE_DEVICES，也支持手动覆盖                                                                                                                            
- 启动前自动检查 torch_npu、MEGATRON_PATH、checkpoint、tokenizer 是否存在                                                                                                       
- 所有参数均可通过环境变量覆盖                                                                                                                                                  
- custom 模式支持传入任意 Megatron-LM 参数 (EXTRA_ARGS)                                                                                                                         
                                                                        