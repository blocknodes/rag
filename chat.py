from qwen_agent.llm import get_chat_model
import sys
if __name__ == "__main__":
    import argparse
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        'model': 'qwen3',
        'model_server': 'http://10.24.9.6:19875/v1',
        'generate_cfg': {
            
        },
    })

    content='''
### step.1 获取预训练模型
#### Code Source
```
link: https://github.com/yangxy/GPEN
branch: master
commit: b611a9f2d05fdc5b82715219ed0f252e42e4d8be
```

#### onnx&torchscript
- 拉取代码至`source_code`目录下
- 将[export.py](./export.py)移动至`source_code/GPEN`目录下
- 修改[gpen_model.py#L690](https://github.com/yangxy/GPEN/blob/main/face_model/gpen_model.py#L690)，在return前添加以下代码：
    ```python
    if len(outs) == 2:
        if outs[1] is None:
            outs = outs[0]

    return outs
    ```
- 执行转换脚本，得到`onnx`和`torchscript`：
    ```python
    python super_resolution/gpen/source_code/GPEN/export.py
    ```
#### Tips
- GPEN默认会将灰度图人脸恢复成彩色图人脸，尺寸没变化

### step.2 准备数据集
- 按论文，取[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集的前1000张图像作为HQ
- 基于[hq2gray.py](../source_code/hq2gray.py)，使用将彩色高清图像转换为灰度图像
- 处理好的数据集
  - 测试彩色HQ图像：[CelebAMask-HQ/GPEN/hq](http://192.168.20.139:8888/vastml/dataset/sr/CelebAMask-HQ/GPEN/hq/?download=zip)
  - 测试灰度HQ图像：[CelebAMask-HQ/GPEN/hq_gray](http://192.168.20.139:8888/vastml/dataset/sr/CelebAMask-HQ/GPEN/hq_gray/?download=zip)
  - 测试灰度HQ图像npz：[CelebAMask-HQ/GPEN/hq_gray_npz](http://192.168.20.139:8888/vastml/dataset/sr/CelebAMask-HQ/GPEN/hq_gray_npz/?download=zip)
  - 测试灰度HQ图像datalist_npz_gray.txt：[CelebAMask-HQ/GPEN/npz_datalist_gray.txt](http://192.168.20.139:8888/vastml/dataset/sr/CelebAMask-HQ/GPEN/npz_datalist_gray.txt)

> Tips
>
> 基于[image2npz.py](../../utils/image2npz.py)，将灰度图像转为npz格式
>

### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型，修改编译配置
    - [official_gpen.yaml](../vacc_code/build/official_gpen.yaml)
    
    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译
    ```bash
    cd gpen
    mkdir workspace
    cd workspace
    vamc compile ../vacc_code/build/official_gpen.yaml
    ```

### step.4 模型推理
1. runmodel
    - 参考：[sample_runmodel.py](../vacc_code/runmodel/sample_runmodel.py)

    ```bash
    python ../vacc_code/runmodel/sample_runmodel.py \
        --file_path /path/to/GPEN/hq_gray/ \
        --model_weight_path deploy_weights/official_gpen_run_model_fp16/  \
        --model_name mod \
        --model_input_name input \
        --model_input_shape 1,3,512,512 \
        --gt_path /path/to/GPEN/hq \
        --save_dir ./runmodel_output 
    ```

2. runstream
    - 参考：[official_vsx_inference.py](../vacc_code/vsx/python/official_vsx_inference.py)
    ```bash
    python ../vacc_code/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/GPEN/hq_gray/ \
        --model_prefix_path deploy_weights/official_gpen_run_stream_fp16/mod \
        --vdsp_params_info ../vacc_code/vdsp_params/official-gpen-vdsp_params.json \
        --hr_image_dir /path/to/GPEN/hq \
        --save_dir ./runstream_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 21.95145274270508, mean ssim: 0.8669498382798001

    # int8
    mean psnr: 22.29287021952575, mean ssim: 0.8196632045742404
    ```

### step.5 性能测试
1. 获取[vamp](../../../docs/doc_vamp.md)工具

3. 性能测试
    - 配置vdsp参数[official-gpen-vdsp_params.json](../vacc_code/vdsp_params/official-gpen-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_gpen_run_stream_int8/mod \
    --vdsp_params ../vacc_code/vdsp_params/official-gpen-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512]
    ```

    <details><summary>点击查看性能测试结果</summary>

    ```
    # fp16
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 95.457
    throughput (qps): 10.7273
    ai utilize (%): 88.2603
    die memory used (MB): 1315.59
    e2e latency (us):
        avg latency: 1572259
        min latency: 94744
        max latency: 1587148
    model latency (us):
        avg latency: 82275
        min latency: 82275
        max latency: 82275

    # int8
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 46.0186
    throughput (qps): 22.2519
    ai utilize (%): 90.3119
    die memory used (MB): 1107.89
    e2e latency (us):
        avg latency: 757924
        min latency: 46906
        max latency: 768466
    model latency (us):
        avg latency: 40586
        min latency: 40586
        max latency: 40586

    # 硬件信息
    Smi version:3.2.1
    SPI production for Bbox mode information of
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    #               Field Name                    Value
    0              FileVersion                       V2
    1                 CardType                  VA1-16G
    2                      S/N             FCA129E00172
    3                 BboxMode              Highperf-AI
    =====================================================================
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    OCLK:       880 MHz    ODSPCLK:    835 MHz    VCLK:       300 MHz    
    ECLK:        20 MHz    DCLK:        20 MHz    VDSPCLK:    900 MHz    
    UCLK:      1067 MHz    V3DCLK:     100 MHz    CCLK:      1000 MHz    
    XSPICLK:     50 MHz    PERCLK:     200 MHz    CEDARCLK:   500 MHz
    ```

    </details>

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`datalist_npz_gray.txt`：
    ```bash
    python ../../utils/image2npz.py \
        --dataset_path GPEN/hq_gray \
        --target_path GPEN/hq_gray_npz \
        --text_path datalist_npz_gray.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_gpen_run_stream_int8/mod \
        --vdsp_params ../vacc_code/vdsp_params/official-gpen-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,512,512] \
        --datalist datalist_npz_gray.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，
    ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
        --gt_dir GPEN/hq \
        --input_npz_path datalist_npz_gray.txt \
        --out_npz_dir outputs/gpen \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

## Tips
- 当前只支持torchscript模型编译（编译最后虽然报段错误，但是模型本身是正确的），onnx模型编译会直接报错
- GPEN有多个模型，可实现人脸恢复、人脸上色等任务
- 仓库提供的face colorization模型为1024尺寸；此处实现，基于FFHQ数据集自训练的512尺寸(耗时7天)，`facegan = FaceGAN(base_dir=".", in_size=512, out_size=512, model='GPEN', channel_multiplier=2, narrow=1,  device="cpu")`
- build参数和原GPEN(facesr)一致

请根据以上内容回答：如何获取GPEN的预训练模型？
'''
    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'system', 'content': "You are a helpful assistant."}, 
                {'role': 'user', 'content': content}]

    response = llm.chat(messages)
    print(response[0].content)