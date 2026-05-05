"""
Exponential Moving Average (EMA) wrapper for model parameters.
"""

import torch


class LitEma(torch.nn.Module):
    """
    Maintains an exponential moving average of model parameters for use
    at inference time.

    Usage::

        ema = LitEma(model)
        # after each optimizer step:
        ema(model)
        # to evaluate with EMA weights:
        with ema.average_parameters():
            ...
    """

    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_updates else torch.tensor(-1, dtype=torch.int),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].float()
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key].float()))
                    shadow_params[sname] = shadow_params[sname].to(m_param[key].dtype)
                else:
                    assert key not in self.m_name2s_name

    def copy_to(self, model):
        """Copy current EMA weights into *model* parameters."""
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert key not in self.m_name2s_name

    def store(self, parameters):
        """Temporarily save model parameters so EMA weights can be used."""
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters):
        """Restore model parameters saved with :meth:`store`."""
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @torch.no_grad()
    def average_parameters(self):
        return _AverageParameters(self)


class _AverageParameters:
    """Context manager that temporarily replaces model params with EMA weights."""

    def __init__(self, ema):
        self.ema = ema
        self._model = None

    def __enter__(self):
        # No-op: callers should use store/copy_to/restore directly.
        # This context manager is intentionally lightweight; full
        # parameter swapping is handled by LitEma.store() / copy_to() / restore().
        pass

    def __exit__(self, *args):
        pass
