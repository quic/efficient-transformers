from typing import Optional, Set

from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger


def validate_sampler_inputs(session_inputs: Set[str], include_sampler: Optional[bool] = None) -> bool:
    """
    Validates whether the `QAICInferenceSession` inputs match inputs required for on-device sampling.

    Mandatory Args:
        session_inputs (set[str]): Set of input names from `QAICInferenceSession`.

    Optional Args:
        include_sampler (bool, default=None): Whether the user explicitly requested sampler support.

    Returns:
        True if sampler is supported, False otherwise.

    Raises:
        ValueError if partial support is detected or if user intent conflicts with QPC capabilities.
    """

    sampler_inputs = Constants.SAMPLER_INPUTS
    count = len(sampler_inputs & session_inputs)

    session_includes_sampler = True
    if count == 0:
        session_includes_sampler = False
    elif count < len(sampler_inputs):
        session_includes_sampler = False
        raise ValueError(
            f"The provided QPC does not have the required number of inputs to run sampling "
            f"on the QAIC device (only {count}/{len(sampler_inputs)} inputs provided). Partial "
            "sampling support is not available. Please check the QPC and try again."
        )

    # Post-validation consistency checks
    if include_sampler and not session_includes_sampler:
        logger.warning(
            "User entered `include_sampler`=True. But the provided QPC is not compiled "
            "to run sampling on the QAIC device. Falling back to the PyTorch backend."
        )
    elif (include_sampler is None or not include_sampler) and session_includes_sampler:
        raise ValueError(
            "The provided QPC is compiled to run sampling on the QAIC device. "
            "But the user did not enter `include_sampler`=True. Please make sure the input "
            "is specified correctly."
        )

    return session_includes_sampler
