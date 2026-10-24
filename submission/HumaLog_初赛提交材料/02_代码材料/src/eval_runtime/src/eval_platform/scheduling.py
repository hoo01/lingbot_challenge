"""Stable, disjoint task groups; all seeds of a group stay in one process."""
def partition(config, count):
    if type(count) is not int or count not in (1, 2):
        raise ValueError('parallelism must be 1 or 2')
    axis = 'tasks' if len(config['tasks']) >= len(config['settings']) else 'settings'
    count = min(count, len(config[axis]))
    return [dict(id=i, tasks=config['tasks'][i::count] if axis == 'tasks' else config['tasks'],
                 settings=config['settings'][i::count] if axis == 'settings' else config['settings'])
            for i in range(count)]
