from praxis_agent.config import ModelConfig
from praxis_agent.context.model_metadata import ModelLimits


def model_config(**values):
    defaults = dict(model="test", base_url="https://example.test/v1", protocol="responses",
                    model_limits=ModelLimits(128000, max_output_tokens=8192), api_key="test")
    defaults.update(values)
    return ModelConfig(**defaults)
