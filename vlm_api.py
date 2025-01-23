import torch
from openai import OpenAI
import base64
import io
from PIL import Image
import numpy as np

class VLMAPINode:
    """
    ComfyUI节点 - VLM API调用
    用于调用各类大语言模型和视觉语言模型的API接口
    支持自定义模型名称、API地址、系统提示词等参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "glm-4"}),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://open.bigmodel.cn/api/paas/v4/"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    CATEGORY = "api"
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    
    FUNCTION = "call_api"
    
    def _encode_image(self, image_tensor):
        """将图像tensor转换为base64编码"""
        try:
            # 将tensor转移到CPU并转换为numpy数组
            image_np = image_tensor.cpu().numpy()
            
            # 打印输入图像的形状和类型，用于调试
            print(f"Input image shape: {image_np.shape}, dtype: {image_np.dtype}")
            
            # 处理不同的输入维度情况
            if len(image_np.shape) == 4:  # (batch, channels, height, width)
                image_np = image_np[0]
            
            # 如果输入是(1, 1, H)格式，需要调整为(H, W)格式
            if len(image_np.shape) == 3 and image_np.shape[0] == 1 and image_np.shape[1] == 1:
                image_np = image_np[0, 0]  # 降维到2D
                # 转换为3通道图像
                image_np = np.stack([image_np, image_np, image_np], axis=-1)
            else:
                # 确保是3通道RGB图像
                if len(image_np.shape) == 3 and image_np.shape[0] == 1:  # 如果是单通道图像
                    image_np = np.repeat(image_np, 3, axis=0)  # 将单通道转换为三通道
                    image_np = np.transpose(image_np, (1, 2, 0))  # 转换为(H, W, C)格式
            
            # 确保值范围在0-255之间
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            print(f"Processed image shape: {image_np.shape}, dtype: {image_np.dtype}")
            
            # 转换为PIL Image
            image = Image.fromarray(image_np)
            
            # 转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"Error in _encode_image: {str(e)}")
            raise
    
    def call_api(self, model_name, api_key, base_url, system_prompt, user_prompt, 
                temperature, top_p, image=None):
        """调用API获取响应"""
        try:
            # 初始化API客户端
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # 如果提供了图像，添加到用户消息中
            if image is not None:
                img_b64 = self._encode_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", 
                         "image_url": {
                             "url": f"data:image/png;base64,{img_b64}"
                         }}
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": user_prompt
                })
            
            # 调用API
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )
            
            # 获取响应文本
            response = completion.choices[0].message.content
            
            return (response,)
            
        except Exception as e:
            return (f"Error: {str(e)}",) 