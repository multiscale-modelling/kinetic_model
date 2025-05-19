class EnergyPlaceholder:
    """
    Placeholder energy‐barrier model.
    Ignores inputs and always returns a fixed set of barrier parameters.
    """
    def __init__(self, pressure: float, temperature: float):
        # store inputs in case you extend this later
        self.pressure = pressure
        self.temperature = temperature

        # static barrier parameters
        self.barrier_params = {
            'C2_ins':  84.382, 'C2_den': 100.151,
            'C2_ads': -73.930, 'C2_des':  73.930,
            'C4_ads': -78.151, 'C4_des':  78.151,
        }

    def get_params(self) -> dict:
        """
        Return the barrier parameters.
        """
        # return a copy to prevent accidental mutation
        return self.barrier_params.copy()

    __call__ = get_params
