
import numpy as np
import pandas as pd

def get_reaction_profile222222222(reaction_profile_type='Ni_dimer_2_paths'):
    if reaction_profile_type == 'Ni_dimer_2_paths':    
        data = {
            'E': [  0.0,  -53,  -69,  
                -46.0, 10, -38, -64,  
                    -113, -125,
                    32, -40, -29, -32 ],
            'type': ['I','I','I',
                    'I','I','I','I',
                    'I', 'I',  
                        'TS','TS', 'TS', 'TS'],
            'state': ['0','1','2',
                    '3','4','5','6', 
                        "3'", "4'",
                        '1,2','2,3', "3',4'", "5,6"],
            'x':    [   0,    1,    2,    
                    3,    4,   5,   6,  
                        3, 5,
                        np.nan, np.nan, np.nan, np.nan]
        }

        df = pd.DataFrame(data)

        # define which intermediates get dashed connectors
        connections = {
            '0': '1',
            '3': '4',
            "3": '4',
            '4': '5',
            "2": "3'",
            "4'": '6'
            # add more as needed...
        }


    return df, connections

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration for different reaction profiles
# ------------------------------------------------------------------------------
PROFILE_CONFIGS = {
    'Ni_dimer_2_paths': {
        'data': {
            'E': [  0.0,  -53,  -69,  
                -46.0, 10, -38, -64,  
                    -113, -125,
                    32, -40, -29, -32 ],
            'type': ['I','I','I',
                    'I','I','I','I',
                    'I', 'I',  
                        'TS','TS', 'TS', 'TS'],
            'state': ['0','1','2',
                    '3','4','5','6', 
                        "3'", "4'",
                        '1,2','2,3', "3',4'", "5,6"],
            'x':    [   0,    1,    2,    
                    3,    4,   5,   6,  
                        3, 5,
                        np.nan, np.nan, np.nan, np.nan]
        },
        'connections': {
            '0': '1',
            '3': '4',
            "3": '4',
            '4': '5',
            "2": "3'",
            "4'": '6'
            # add more as needed...
        }
    },

    # you can add more profiles here under new keys
}

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def list_available_profiles():
    """Return the list of supported reaction_profile_type keys."""
    return list(PROFILE_CONFIGS.keys())

def _validate_connections(connections, available_states):
    """
    Ensure that every key and every target in `connections` exists in available_states.
    Raise KeyError with a clear message if anything is invalid.
    """
    bad_keys = [k for k in connections if k not in available_states]
    if bad_keys:
        raise KeyError(
            f"Connection key(s) {bad_keys} not found. "
            f"Available states: {sorted(available_states)}"
        )

    # flatten targets into a list
    bad_targets = []
    for k, targets in connections.items():
        if not isinstance(targets, (list, tuple, set)):
            targets_list = [targets]
        else:
            targets_list = targets
        for t in targets_list:
            if t not in available_states:
                bad_targets.append((k, t))

    if bad_targets:
        msgs = [f"'{t}' (from key '{k}')" for k, t in bad_targets]
        raise KeyError(
            f"Connection target(s) {msgs} not found. "
            f"Available states: {sorted(available_states)}"
        )

def get_reaction_profile(reaction_profile_type='Ni_dimer_2_paths'):
    """
    Retrieve the DataFrame and connections dict for the given profile type.

    Raises:
      ValueError if the profile_type is unknown.
      KeyError   if the stored connections reference non-existent states.
    """
    if reaction_profile_type not in PROFILE_CONFIGS:
        avail = ', '.join(list_available_profiles())
        raise ValueError(
            f"Unknown reaction_profile_type '{reaction_profile_type}'. "
            f"Available types: {avail}"
        )

    cfg = PROFILE_CONFIGS[reaction_profile_type]
    df = pd.DataFrame(cfg['data'])
    connections = cfg['connections']

    # Validate that all connection keys/targets exist in df['state']
    available_states = set(df['state'].astype(str).unique())
    _validate_connections(connections, available_states)

    return df, connections

# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df, connections = get_reaction_profile('Ni_dimer_2_paths')
        print("DataFrame:\n", df, "\n")
        print("Connections:\n", connections)
    except (ValueError, KeyError) as e:
        print("Error:", e)
