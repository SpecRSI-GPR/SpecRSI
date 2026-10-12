"""One-chat token IDs, independent of tokenizer return-container defaults."""
from collections.abc import Mapping
import inspect
from numbers import Integral


def chat_token_ids(tokenizer, messages, **options):
    """Render one complete chat into flat integer IDs, without truncation.

    Transformers 5.17 defaults apply_chat_template(return_dict=True), whereas
    older supported tokenizers returned a list. Generation and token metering
    must consume the identical complete input_ids, never dictionary keys or
    the number of fields in a BatchEncoding. Explicitly request list output,
    then normalize a retained dictionary or single-batch tensor from wrappers.
    Legacy callables with a closed signature need not accept return_dict.
    """
    if options.get('tokenize', True) is not True:
        raise ValueError('chat_token_ids requires tokenized output')
    options = {**options, 'tokenize': True, 'return_dict': False}
    method = tokenizer.apply_chat_template
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None and 'return_dict' not in parameters and not any(
            value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
        options.pop('return_dict')
    values = method(messages, **options)
    if isinstance(values, Mapping):
        if 'input_ids' not in values:
            raise ValueError('Chat encoding must contain input_ids')
        values = values['input_ids']
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError('Chat encoding must be a token sequence')
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError('Expected exactly one chat, not a batch of conversations')
        values = values[0]
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values):
        raise ValueError('Chat input_ids must be a flat nonnegative integer sequence')
    return [int(value) for value in values]
