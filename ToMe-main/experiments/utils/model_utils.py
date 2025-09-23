def patch_model_with_strategy(model, strategy):
    """为模型应用特定的相似度策略"""
    if hasattr(model, '_tome_info'):
        model._tome_info['similarity_strategy'] = strategy
