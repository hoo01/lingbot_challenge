"""JSON encoding for static simulator metadata, never policy inputs."""
import math


def joint_limits_for_json(limits):
    """Keep finite bounds numeric and represent legitimate unbounded limits.

    SAPIEN reports continuous joints with [-inf, inf]. Only joint-limit
    metadata uses this conversion; NaN limits and nonfinite dynamic states
    or actions must still fail strict trajectory serialization.
    """
    result = []
    for lower, upper in limits:
        pair = []
        for bound in (lower, upper):
            value = float(bound)
            if math.isnan(value):
                raise ValueError('NaN joint limit is not valid metadata')
            pair.append(('+Infinity' if value > 0 else '-Infinity')
                        if math.isinf(value) else value)
        result.append(pair)
    return result
