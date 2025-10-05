
"""
安全的Prompt工具函数 - 避免LangChain模板变量冲突
"""

def safe_json_format(json_str: str) -> str:
    """
    安全地在prompt中使用JSON格式，自动转义花括号
    
    Args:
        json_str: JSON格式字符串
        
    Returns:
        转义后的JSON字符串，可安全用于LangChain prompt
    """
    # 如果已经转义过，就不再处理
    if json_str.count('{{') == json_str.count('}}') and '{{' in json_str:
        return json_str
    
    # 转义单花括号为双花括号
    return json_str.replace('{', '{{').replace('}', '}}')

def create_safe_response_format(format_dict: dict) -> str:
    """
    创建安全的回答格式提示
    
    Args:
        format_dict: 格式字典，如 {"score": "0-1之间的数值", "reasoning": "理由"}
        
    Returns:
        安全的格式提示字符串
    """
    import json
    json_str = json.dumps(format_dict, ensure_ascii=False, indent=4)
    return f"""回答格式：
{safe_json_format(json_str)}
"""

# 使用示例：
# format_prompt = create_safe_response_format({
#     "score": "0.0-1.0之间的数值",
#     "reasoning": "评分理由"
# })
