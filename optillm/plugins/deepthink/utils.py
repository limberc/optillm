import logging

logger = logging.getLogger(__name__)

def create_completion(client, **kwargs):
    """
    Helper function to call client.chat.completions.create with retry logic
    for max_tokens vs max_completion_tokens compatibility.
    """
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        # Check for max_tokens error
        # OpenAI error message usually contains: "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
        error_msg = str(e).lower()
        if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
            # Swap parameters
            if 'max_tokens' in kwargs:
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
                logger.info("Retrying with max_completion_tokens instead of max_tokens")
                return client.chat.completions.create(**kwargs)
        raise e
