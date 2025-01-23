import torch
from openai import OpenAI
import base64
import io
from PIL import Image

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
        # 确保图像tensor格式正确
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # 转换为PIL Image
        image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0))
        
        # 转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
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