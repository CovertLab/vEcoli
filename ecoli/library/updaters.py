'''
============================
Registry of Inverse Updaters
============================

You should interpret words and phrases that appear fully capitalized in
this document as described in :rfc:`2119`. Here is a brief summary of
the RFC:

* "MUST" indicates absolute requirements. Vivarium may not work
  correctly if you don't follow these.
* "SHOULD" indicates strong suggestions. You might have a valid reason
  for deviating from them, but be careful that you understand the
  ramifications.
* "MAY" indicates truly optional features that you can include or
  exclude as you wish.

----------------
Inverse Updaters
----------------

Inverse updaters accept a final and an initial state, and they attempt
to compute an update that could be given to a particular :term:`updater`
to get from the initial state to the final state.

Inverse Updater API
===================

An inverse updater MUST be registered with the same name as its
associated updater. The function MUST accept exactly 2 positional
arguments: the initial state and the final state. The function SHOULD
not accept any other parameters. The function MUST return only the
update needed to get from the initial state to the final state. The
function SHOULD have a name that matches its associated updater
function, only prefixed with ``inverse_``. Inverse updaters MUST return
an empty dictionary only if the update can be ignored.
'''

import numpy as np
from vivarium.core.registry import Registry


inverse_updater_registry = Registry()


def inverse_update_set(initial_state, final_state):
    if np.all(initial_state == final_state):
        return {}
    return final_state


def inverse_update_null(initial_state, final_state):
    '''The null updater ignores the update.'''
    return {}


def inverse_update_accumulate(initial_state, final_state):
    if np.all(initial_state == final_state):
        return {}
    return final_state - initial_state


def inverse_update_nonnegative_accumulate(initial_state, final_state):
    if np.all(initial_state == final_state):
        return {}
    return final_state - initial_state


def _reverse_dict_update(initial, final):
    '''
    Return an update such that ``initial.update(update) == final``.
    '''
    update = {}
    for key, val in final.items():
        if key in initial and val != initial[key]:
            # We need to replace initial[key] with final[key].
            update[key] = val
        elif key not in initial:
            # We need to add key (with value final[key]) to initial.
            update[key] = val
    return update


#: Special signal used by _reverse_deep_merge to communicate that no
#: updates are needed. Using this signal reduces the update size by
#: eliminating subtrees of empty updates.
_NO_UPDATE_NEEDED = 'No update needed'


def _reverse_deep_merge(initial, final):
    if not isinstance(initial, dict):
        if initial != final:
            return final
        return _NO_UPDATE_NEEDED
    update = {}
    for key in final:
        if key not in initial:
            update[key] = final[key]
        else:
            sub_update = _reverse_deep_merge(
                initial[key], final[key])
            if sub_update is not _NO_UPDATE_NEEDED:
                update[key] = sub_update
    return update or _NO_UPDATE_NEEDED


def inverse_update_merge(initial_state, final_state):
    update = _reverse_deep_merge(initial_state, final_state)
    return {} if update is _NO_UPDATE_NEEDED else update


def inverse_update_dictionary(initial_state, final_state):
    update = {}

    for key in final_state.keys():
        if key not in initial_state:
            continue
        if initial_state[key] != final_state[key]:
            update[key] = _reverse_dict_update(
                initial_state[key], final_state[key])

    deleted = initial_state.keys() - final_state.keys()
    if deleted:
        update['_delete'] = list(deleted)

    added = final_state.keys() - initial_state.keys()
    if added:
        # TODO: How do I figure out the path, especially when that path
        # is beyond the state seen by the updater?
        pass
    return update
