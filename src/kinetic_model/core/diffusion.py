from kinetic_model.core.reactor import Reaction

class Reaction_Diffusion_old(Reaction):
    def __init__(self, reactions, capacity=None, fixed_concentrations=None) -> None:
        super().__init__(reactions, fixed_concentrations)
        self.capacity = capacity
        if capacity is not None:
            self._normalize_capacity()

    def parse_diffusion(self, reaction_str):
        # parse diffusion reaction
        is_back = False
        if '<D>' in reaction_str:
            reactants_str, products_str = reaction_str.split('<D>')
            is_back = True
        else:
            reactants_str, products_str = reaction_str.split('D>')
        
        reactants = [r.strip() for r in reactants_str.split('+')]
        products = [p.strip() for p in products_str.split('+')]
        
        return reactants, products, is_back
    
    def _normalize_capacity(self):
        norm_factor = sum(self.capacity.values())
        for key in self.capacity:
            self.capacity[key] = self.capacity[key] / norm_factor

    def parse_reaction(self, reaction_str):
        if ('<->' in reaction_str) or ('->' in reaction_str):
            return super().parse_reaction(reaction_str), False
        else:
            return self.parse_diffusion(reaction_str), True

    def create_reaction_dict(self):
        result = {}
        
        for reaction in self.reactions:
            (reactants, products, is_back), is_diffusion = self.parse_reaction(reaction[0])
            for reactant in reactants:
                if reactant not in result:
                    result[reactant] = []
                coefficient = 1/self.capacity[reactant] if is_diffusion else 1
                result[reactant].append((tuple(reactants), -reaction[1] * coefficient))
                if is_back:
                    coefficient = 1/self.capacity[products[0]] if is_diffusion else 1
                    result[reactant].append((tuple(products), reaction[2] * coefficient))
            for product in products:
                if product not in result:
                    result[product] = []
                
                coefficient = 1/self.capacity[reactant] if is_diffusion else 1
                result[product].append((tuple(reactants), reaction[1] * coefficient))
                if is_back:
                    coefficient = 1/self.capacity[product] if is_diffusion else 1
                    result[product].append((tuple(products), -reaction[2] * coefficient))

        self.reaction_dict = result  
        self.sorted_keys = sorted(self.reaction_dict.keys())   



class Reaction_Diffusion(Reaction):
    """
    Reaction + compartmental diffusion with capacity-weighted flux.
    Diffusion step 'X@A <D> X@B' is interpreted as:
        dC_A/dt = -(k_f/Cap_A)*C_A + (k_b/Cap_B)*C_B
        dC_B/dt =  (k_f/Cap_A)*C_A - (k_b/Cap_B)*C_B
    One-way 'X@A D> X@B' drops the reverse terms.
    NOTE: capacities are used as-given (no global normalization).
    """

    def __init__(self, reactions, capacity=None, fixed_concentrations=None) -> None:
        super().__init__(reactions, fixed_concentrations)
        self.capacity = dict(capacity or {})  # species -> capacity (e.g., pore vol*sites)
        # IMPORTANT: no normalization here

    def parse_diffusion(self, reaction_str):
        # Returns (reactants, products, is_back) where each side is a single species list
        if '<D>' in reaction_str:
            reactants_str, products_str = reaction_str.split('<D>')
            is_back = True
        else:
            reactants_str, products_str = reaction_str.split('D>')
            is_back = False
        reactants = [r.strip() for r in reactants_str.split('+')]   # expect len==1
        products  = [p.strip() for p in products_str.split('+')]    # expect len==1
        return reactants, products, is_back

    def parse_reaction(self, reaction_str):
        # Return ((reactants, products, is_back), is_diffusion)
        if ('<->' in reaction_str) or ('->' in reaction_str):
            return super().parse_reaction(reaction_str), False
        else:
            return self.parse_diffusion(reaction_str), True

    def _cap(self, species: str) -> float:
        # Safe getter: default capacity 1.0 if not provided
        c = self.capacity.get(species, 1.0)
        if c <= 0.0:
            raise ValueError(f"Capacity for '{species}' must be positive; got {c}.")
        return c

    def create_reaction_dict(self):
        result = {}

        def ensure_key(s):
            if s not in result:
                result[s] = []

        for reaction in self.reactions:
            (reactants, products, is_back), is_diffusion = self.parse_reaction(reaction[0])

            if is_diffusion:
                # Expect exactly one species on each side for diffusion edges
                src = reactants[0]
                dst = products[0]
                kf  = reaction[1]
                kb  = reaction[2] if is_back and len(reaction) > 2 else 0.0

                cap_src = self._cap(src)
                cap_dst = self._cap(dst)

                # Source balance: - (kf/Cap_src)*C_src   + (kb/Cap_dst)*C_dst
                ensure_key(src)
                result[src].append(((src,), -kf / cap_src))
                if is_back and kb != 0.0:
                    result[src].append(((dst,), +kb / cap_dst))

                # Destination balance: + (kf/Cap_src)*C_src   - (kb/Cap_dst)*C_dst
                ensure_key(dst)
                result[dst].append(((src,), +kf / cap_src))
                if is_back and kb != 0.0:
                    result[dst].append(((dst,), -kb / cap_dst))

            else:
                # Chemical reaction: same as base, no capacity scaling
                reactants_list, products_list, is_back = reactants, products, is_back
                kf = reaction[1]
                kb = reaction[2] if is_back and len(reaction) > 2 else None

                # Consume reactants / produce products (forward)
                for s in reactants_list:
                    ensure_key(s)
                    result[s].append((tuple(reactants_list), -kf))
                for s in products_list:
                    ensure_key(s)
                    result[s].append((tuple(reactants_list), +kf))

                # Reverse if applicable
                if is_back and kb is not None:
                    for s in reactants_list:
                        result[s].append((tuple(products_list), +kb))
                    for s in products_list:
                        result[s].append((tuple(products_list), -kb))

        self.reaction_dict = result
        self.sorted_keys = sorted(self.reaction_dict.keys())
