from . import simple_spread
from . import simple_reference
from . import simple_speaker_listener
from . import simple
SCENARIOS = {
    'simple_spread': simple_spread.Scenario,
    'simple_reference': simple_reference.Scenario,
    'simple_speaker_listener': simple_speaker_listener.Scenario,
    'simple': simple.Scenario,
} 