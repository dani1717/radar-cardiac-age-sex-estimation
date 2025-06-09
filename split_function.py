
import random
import numpy as np

#%% Person data

person_data = {
    1: {'age': 24, 'sex': 'F', 'height': 166, 'weight': 63},
    2: {'age': 38, 'sex': 'F', 'height': 161, 'weight': 49},
    3: {'age': 25, 'sex': 'M', 'height': 187, 'weight': 82},
    4: {'age': 28, 'sex': 'F', 'height': 178, 'weight': 59},
    5: {'age': 27, 'sex': 'F', 'height': 173, 'weight': 93},
    6: {'age': 49, 'sex': 'F', 'height': 172, 'weight': 63},
    7: {'age': 24, 'sex': 'F', 'height': 187, 'weight': 80},
    8: {'age': 24, 'sex': 'M', 'height': 182, 'weight': 77},
    9: {'age': 40, 'sex': 'M', 'height': 184, 'weight': 74},
    10: {'age': 24, 'sex': 'M', 'height': 186, 'weight': 78},
    11: {'age': 21, 'sex': 'F', 'height': 165, 'weight': 55},
    12: {'age': 24, 'sex': 'M', 'height': 193, 'weight': 86},
    13: {'age': 49, 'sex': 'F', 'height': 173, 'weight': 62},
    14: {'age': 35, 'sex': 'F', 'height': 153, 'weight': 44},
    15: {'age': 27, 'sex': 'M', 'height': 182, 'weight': 78},
    16: {'age': 42, 'sex': 'F', 'height': 165, 'weight': 61},
    17: {'age': 49, 'sex': 'F', 'height': 167, 'weight': 85},
    18: {'age': 28, 'sex': 'M', 'height': 165, 'weight': 57},
    19: {'age': 27, 'sex': 'F', 'height': 166, 'weight': 59},
    20: {'age': 23, 'sex': 'F', 'height': 172, 'weight': 68},
    21: {'age': 24, 'sex': 'M', 'height': 187, 'weight': 85},
    22: {'age': 61, 'sex': 'F', 'height': 178, 'weight': 90},
    23: {'age': 27, 'sex': 'M', 'height': 186, 'weight': 69},
    24: {'age': 21, 'sex': 'F', 'height': 165, 'weight': 65},
    25: {'age': 26, 'sex': 'M', 'height': 183, 'weight': 82},
    26: {'age': 31, 'sex': 'F', 'height': 160, 'weight': 51},
    27: {'age': 24, 'sex': 'M', 'height': 187, 'weight': 83},
    28: {'age': 29, 'sex': 'M', 'height': 190, 'weight': 94},
    29: {'age': 25, 'sex': 'M', 'height': 186, 'weight': 82},
    30: {'age': 26, 'sex': 'M', 'height': 172, 'weight': 93}
}



#%% Functions for splitting dataset by person

def split_train_and_test_by_person(frames, windows, prop_train, person_column='person', age_binBounds=None, seed=None):
    """
    Split dataset into training and testing sets by person, ensuring no person appears in both sets.
    
    Args:
        frames (pd.DataFrame): DataFrame containing frame metadata including person ID.
        windows (pd.DataFrame): DataFrame containing window metadata indexed by 'window'.
        prop_train (float): Proportion of persons to include in the training set.
        person_column (str): Column name in frames that identifies the person.
        age_binBounds (list or None): If None, balance split by sex; else balance by age groups defined by these bounds.
        seed (int or None): Seed for random number generator for reproducibility.
        
    Returns:
        frames_train (pd.DataFrame): Subset of frames for training.
        frames_test (pd.DataFrame): Subset of frames for testing.
    """

    selected_windows = windows[windows.index.isin(frames['window'].unique())]
    persons = selected_windows[person_column].unique()
    n_persons = len(persons)
    n_persons_train = int(round(n_persons * prop_train, 0))
    n_persons_test = n_persons - n_persons_train

    if age_binBounds is None:
        # Balance split by sex
        persons_test_original = sample_balanced_persons_by_sex(persons, n_persons_test, seed=seed)
        sampling_type = 'sex'
    else:
        # Balance split by age groups
        persons_test_original = sample_balanced_persons_by_age(persons, n_persons_test, age_binBounds, seed=seed)
        sampling_type = 'age'

    frames_test = frames[frames[person_column].isin(persons_test_original)]
    frames_train = frames[~frames[person_column].isin(persons_test_original)]
    persons_test_mapped = frames_test['person'].unique().tolist()

    # Summary printout
    if sampling_type == 'sex':
        sex_count = {'M': 0, 'F': 0}
        for person in persons_test_original:
            sex = person_data[person]['sex']
            sex_count[sex] += 1
        print(f'Split: {sex_count["M"]} males and {sex_count["F"]} females were separated for testing '
              f'(persons {persons_test_mapped}) and {n_persons_train} persons for training.')
    else:
        # Collect ages of persons in test set
        ages = sorted([person_data[person]['age'] for person in persons_test_original])
        print(f'Split by age: Ages in test set: {ages} (persons {persons_test_mapped}) and {n_persons_train} persons for training.')

    return frames_train, frames_test


def sample_balanced_persons_by_sex(persons, n_persons, seed=None):
    """
    Sample persons to create a balanced test set by sex.
    
    Args:
        persons (array-like): List or array of person IDs to sample from.
        n_persons (int): Number of persons to sample.
        seed (int or None): Seed for random number generator.
        
    Returns:
        sampled_persons (list): List of sampled person IDs balanced by sex.
    """

    if seed is not None:
        previous_seed = random.getstate()
        random.seed(seed)

    males = [person for person in person_data if person_data[person]['sex'] == 'M' and person in persons]
    females = [person for person in person_data if person_data[person]['sex'] == 'F' and person in persons]

    n_males = n_persons // 2
    n_females = n_persons - n_males

    sampled_males = random.sample(males, min(n_males, len(males)))
    sampled_females = random.sample(females, min(n_females, len(females)))

    sampled_persons = sampled_males + sampled_females
    random.shuffle(sampled_persons)

    if seed is not None:
        random.setstate(previous_seed)

    return sampled_persons


def sample_balanced_persons_by_age(persons, n_persons, age_binBounds, seed=None):
    """
    Sample persons to create a balanced test set by age groups.
    
    Args:
        persons (array-like): List or array of person IDs to sample from.
        n_persons (int): Number of persons to sample.
        age_binBounds (list): List of age boundaries defining age groups.
        seed (int or None): Seed for random number generator.
        
    Returns:
        sampled_persons (list): List of sampled person IDs balanced by age groups.
    """

    if seed is not None:
        previous_seed = random.getstate()
        random.seed(seed)

    # Import age grouping function
    from utils.dataset_organization import age_to_group

    # Assign age groups to all persons
    for person_id, data in person_data.items():
        data['ageGroup'] = age_to_group(data['age'], age_binBounds)

    ageGroups = {data['ageGroup'] for data in person_data.values()}
    n_groups = len(ageGroups)

    # Group persons by age group
    people_by_ageGroup = {ageGroup: [] for ageGroup in ageGroups}
    for person_id, data in person_data.items():
        if person_id in persons:
            people_by_ageGroup[data['ageGroup']].append(person_id)

    # Calculate sample sizes per group
    indices = np.arange(n_persons)
    splits = np.array_split(indices, n_groups)
    group_sizes = {ageGroup: len(split) for ageGroup, split in zip(ageGroups, splits)}

    # Sample persons per age group
    sampled_persons_by_ageGroup = {}
    for ageGroup in ageGroups:
        available = people_by_ageGroup.get(ageGroup, [])
        n_sample = group_sizes.get(ageGroup, 0)
        sampled_persons_by_ageGroup[ageGroup] = random.sample(available, min(n_sample, len(available)))

    # Flatten list and shuffle
    sampled_persons = [person for group in sampled_persons_by_ageGroup.values() for person in group]
    random.shuffle(sampled_persons)

    if seed is not None:
        random.setstate(previous_seed)

    return sampled_persons


