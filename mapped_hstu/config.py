"""Validate a single learned-versus-frozen mapping intervention."""

from content_hstu.config import ROOT, resolved, validate_content_config as validate_original


def validate_content_config(config, reference):
    """Reuse matched budgets while forbidding ID residuals and extra interventions."""
    if config.get('representation') not in {'map-frozen', 'map-learned'}:
        raise ValueError('Expected map-frozen or map-learned')
    if config.get('fusion_weight') != 0:
        raise ValueError('Shared mapping has no ID fusion')
    adapted = dict(config, representation='title-init')
    validate_original(adapted, reference)
