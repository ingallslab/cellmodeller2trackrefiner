import numpy as np
import pickle
import CellModeller
from simOutputProcessing.scripts.neighborsFinding import neighbor_finders
import pandas as pd
import glob


def reassign_id(dataframe, parent_image_number_col, parent_object_number_col):

    dataframe['index'] = dataframe.index
    dataframe['checked'] = False
    dataframe["divideFlag"] = False
    dataframe['id'] = np.nan
    dataframe['LifeHistory'] = np.nan
    dataframe['CellAge'] = np.nan

    cond1 = dataframe[parent_image_number_col] == 0
    cond2 = dataframe['ImageNumber'] > 1
    cond3 = dataframe['ImageNumber'] == 1

    dataframe.loc[cond1 & cond2, ['checked']] = True

    dataframe.loc[cond1 & cond3, ['checked']] = True

    dataframe.loc[cond1, 'id'] = dataframe.loc[cond1, 'index'].values + 1

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = dataframe.merge(dataframe, left_on=[parent_image_number_col, parent_object_number_col],
                                right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division = merged_df[merged_df.duplicated(subset='index_1', keep=False)][['index_1']].copy()

    dataframe.loc[division['index_1'].unique(), "divideFlag"] = True

    # other bacteria
    other_bac_df = dataframe.loc[~ dataframe['checked']]

    temp_df = dataframe.copy()
    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    id_list = []

    same_bac_dict = {}

    last_bac_id = dataframe['id'].max() + 1

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if str(int(parent_img_num)) + '_' + str(parent_obj_num) not in same_bac_dict.keys():
            source_link = dataframe.iloc[bac_index_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]]

            # life history continues
            source_bac_id = source_link['id']
            division_stat = source_link['divideFlag']

        else:
            source_bac_id, division_stat = same_bac_dict[f"{int(parent_img_num)}_{int(parent_obj_num)}"]

        if division_stat:
            # division occurs
            new_bac_id = last_bac_id
            last_bac_id += 1
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [new_bac_id, row['divideFlag']]

            id_list.append(new_bac_id)

        else:
            id_list.append(source_bac_id)

            # same bacteria
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [source_bac_id, row['divideFlag']]

    dataframe.loc[other_bac_df.index, 'id'] = id_list

    dataframe['LifeHistory'] = dataframe.groupby('id')['id'].transform('size')

    # set age
    dataframe['CellAge'] = dataframe.groupby('id').cumcount() + 1

    dataframe['index'] = dataframe.index.values

    dataframe = dataframe.drop(columns=['checked', 'divideFlag'])

    return dataframe


def assign_label_from_nearest_disappeared_bacterium(current_position, previous_positions, disappeared_ids,
                                                    labels_dict, last_occurrence_map):
    """
    Assigns a label to a new bacterium by finding the closest disappeared bacterium
    from the previous time step (i.e., a 'grandmother').

    Parameters:
    - current_position (np.ndarray): 1D array representing the (x, y) position of the new bacterium.
    - previous_positions (np.ndarray): 2D array of shape (n, 2) with (x, y) positions of disappeared bacteria.
    - disappeared_ids (list): List of IDs corresponding to `previous_positions`.
    - labels_dict (dict): Dictionary mapping bacterium IDs to their assigned labels.
    - last_occurrence_map (dict): Dictionary mapping bacterium IDs to their last index in the features dict.

    Returns:
    - label (int): Assigned label from the closest disappeared bacterium.
    - last_index (int): Index of the last appearance of that bacterium in the features dict.
    """
    distances = np.linalg.norm(previous_positions - current_position, axis=1)
    closest_idx = np.argmin(distances)
    nearest_id = disappeared_ids[closest_idx]
    label = labels_dict[nearest_id]
    last_index = last_occurrence_map[nearest_id]
    return label, last_index


def extract_bacterial_features(current_time_step_num, current_time_step_bac, prev_time_step_bac, bac_features_dict,
                               bacteria_labels_dict, neighbor_records, cell_type_mapping, use_grandmother_as_parent):
    """
    Extracts and updates bacterial features for a specific time step from CellModeller simulation data.

    Parameters:
    - current_time_step_num (int): The current simulation time step number.
    - current_time_step_bac (dict): Dictionary containing 'cellStates' and 'lineage' at the current step.
    - prev_time_step_bac (dict): Dictionary containing 'cellStates' and 'lineage' at the previous step.
    - bac_features_dict (dict): Dictionary accumulating features across time steps.
    - bacteria_labels_dict (dict): Mapping from bacteria IDs to assigned tracking labels.
    - neighbor_records (list): List to accumulate neighboring bacteria relationships.
    - cell_type_mapping (dict): Mapping of cell type names to integer IDs.
    - use_grandmother_as_parent (bool):
        If True, approximates the parent bacterium by selecting the nearest disappeared bacterium
        from the previous time step. Useful when large time step gaps cause the actual parent
        to no longer appear in the previous step.

    Returns:
    - bac_features_dict (dict): Updated features dictionary.
    - bacteria_labels_dict (dict): Updated bacteria label mapping.
    - neighbor_records (list): Updated list of neighbor records.
    """

    # bacteria information
    cs = current_time_step_bac['cellStates']
    cs_prev_time_step = prev_time_step_bac['cellStates']

    # Cellmodeller pickle file structure:
    # ['cellStates', 'stepNum', 'lineage', 'moduleStr', 'moduleName']
    # important keys: 1: 'cellStates' (type: dictionary)  2: 'lineage' (dictionary)
    # 'cellStates' dictionary keys: bacteria id
    # 'lineage' dictionary: daughter id: parent id

    # find Bacteria whose life has ended.
    if len(cs_prev_time_step) > 0:
        bac_ids_prev_time_step = set([cs_prev_time_step[it].id for it in cs_prev_time_step.keys()])
    else:
        bac_ids_prev_time_step = set()
    bac_ids_current_time_step = set([cs[it].id for it in cs.keys()])
    life_ended_bacteria_in_previous_time_step = list(bac_ids_prev_time_step - bac_ids_current_time_step)

    # now center coordinate of them
    if len(life_ended_bacteria_in_previous_time_step) > 0:
        life_ended_bac_center_pos_in_previous_time_step = np.array([cs_prev_time_step[b_id].pos[:2] for b_id in
                                                                    life_ended_bacteria_in_previous_time_step])
    else:
        life_ended_bac_center_pos_in_previous_time_step = np.array([])

    bacteria_ids = np.array(bac_features_dict['id'])
    # Get last occurrence of each unique id:
    # Reverse the array and use return_index on it
    unique_bac_ids, reverse_bac_idx = np.unique(bacteria_ids[::-1], return_index=True)

    # Convert reverse indices to forward indices (relative to original)
    bac_last_indices = len(bacteria_ids) - 1 - reverse_bac_idx

    # Build a mapping: id → last index
    bac_id_to_last_idx = dict(zip(unique_bac_ids, bac_last_indices))

    if len(bacteria_labels_dict) > 0:
        max_bacteria_labels = max(bacteria_labels_dict.values()) + 1
    else:
        max_bacteria_labels = 1

    this_time_step_bac_parent_image_number = []
    this_time_step_bac_parent_obj_number = []
    this_time_step_bac_label = []
    this_time_step_bac_cell_age = []

    bac_it = cs.keys()
    prev_time_steps_bac_ids = bacteria_labels_dict.keys()

    for this_bac_key, this_bac_features in cs.items():
        # find neighbors
        this_bac_id = this_bac_features.id
        this_bac_neighbours = this_bac_features.neighbours
        if len(this_bac_neighbours) > 0:
            neighbor_records.extend(
                [(current_time_step_num, this_bac_id, neighbor_id) for neighbor_id in this_bac_neighbours])

        if this_bac_features.id in prev_time_steps_bac_ids:  # it means: Life has continued for bacterium

            # last occurrence of element in list
            last_occurrence_index_in_list = bac_id_to_last_idx[this_bac_features.id]

            # cell age
            this_time_step_bac_cell_age.append(bac_features_dict['CellAge'][last_occurrence_index_in_list] + 1)

            # parent information
            this_time_step_bac_parent_image_number.append(
                bac_features_dict['ImageNumber'][last_occurrence_index_in_list])

            this_time_step_bac_parent_obj_number.append(
                bac_features_dict['ObjectNumber'][last_occurrence_index_in_list])

            # assign label
            this_time_step_bac_label.append(bacteria_labels_dict[this_bac_features.id])

        else:  # it means: A bacterium has been born or a cell division has taken place
            if current_time_step_num == 1:  # birth

                if len(bacteria_labels_dict) == 0:
                    this_bacterium_label = max_bacteria_labels
                    max_bacteria_labels += 1
                else:
                    this_bacterium_label = max_bacteria_labels
                    max_bacteria_labels += 1

                # cell age
                this_time_step_bac_cell_age.append(0)
                # parent information
                this_time_step_bac_parent_image_number.append(0)
                this_time_step_bac_parent_obj_number.append(0)

            else:
                # find parent id
                if current_time_step_bac['lineage'][this_bac_features.id]:  # parent bacteria has been found
                    parent_id = current_time_step_bac['lineage'][this_bac_features.id]
                    # assign label
                    if parent_id in prev_time_steps_bac_ids:
                        this_bacterium_label = bacteria_labels_dict[parent_id]
                        # last occurrence of element in list
                        last_occurrence_index_in_list = bac_id_to_last_idx[parent_id]
                    else:
                        '''
                            Find the nearest bacterium from the previous timestep that does not exist in the current
                             timestep (since the founded bacterium is the grandmother of the daughter)                    
                        '''
                        if len(life_ended_bacteria_in_previous_time_step) > 0 and use_grandmother_as_parent:

                            # calculate distance
                            this_bacterium_center = np.array([this_bac_features.pos[0], this_bac_features.pos[1]])
                            this_bacterium_label, last_occurrence_index_in_list = \
                                assign_label_from_nearest_disappeared_bacterium(
                                    this_bacterium_center, life_ended_bac_center_pos_in_previous_time_step,
                                    life_ended_bacteria_in_previous_time_step, bacteria_labels_dict,
                                    bac_id_to_last_idx)
                        else:
                            this_bacterium_label = max_bacteria_labels
                            max_bacteria_labels += 1
                            last_occurrence_index_in_list = -1

                else:
                    '''
                        Find the nearest bacterium from the previous timestep that does not exist in the 
                        current timestep (since the founded bacterium is the grandmother of the daughter)                    
                    '''

                    if use_grandmother_as_parent:
                        # calculate distance
                        this_bacterium_center = np.array((this_bac_features.pos[0], this_bac_features.pos[1]))
                        this_bacterium_label, last_occurrence_index_in_list = \
                            assign_label_from_nearest_disappeared_bacterium(
                                this_bacterium_center, life_ended_bac_center_pos_in_previous_time_step,
                                life_ended_bacteria_in_previous_time_step, bacteria_labels_dict,
                                bac_id_to_last_idx)
                    else:
                        this_bacterium_label = max_bacteria_labels
                        max_bacteria_labels += 1
                        last_occurrence_index_in_list = -1

                if last_occurrence_index_in_list != -1:
                    # cell age
                    this_time_step_bac_cell_age.append(0)

                    # parent information
                    this_time_step_bac_parent_image_number.append(
                        bac_features_dict['ImageNumber'][last_occurrence_index_in_list])
                    this_time_step_bac_parent_obj_number.append(
                        bac_features_dict['ObjectNumber'][last_occurrence_index_in_list])
                else:
                    # cell age
                    this_time_step_bac_cell_age.append(0)

                    # parent information
                    this_time_step_bac_parent_image_number.append(0)
                    this_time_step_bac_parent_obj_number.append(0)

            # assign label
            this_time_step_bac_label.append(this_bacterium_label)
            # add new key: value to dictionary
            bacteria_labels_dict[this_bac_features.id] = this_bacterium_label

    bac_features_dict['ImageName'].extend([current_time_step_bac['stepNum']] * len(cs))
    bac_features_dict['ImageNumber'].extend([current_time_step_num] * len(cs))
    bac_features_dict['ObjectNumber'].extend(range(1, len(cs) + 1))
    bac_features_dict['AreaShape_Center_X'].extend([cs[it].pos[0] for it in bac_it])
    bac_features_dict['AreaShape_Center_Y'].extend([cs[it].pos[1] for it in bac_it])
    bac_features_dict['AreaShape_MajorAxisLength'].extend([cs[it].length for it in bac_it])
    bac_features_dict['AreaShape_MinorAxisLength'].extend([cs[it].radius for it in bac_it])

    bac_features_dict['AreaShape_Orientation'].extend([np.arctan2(cs[it].dir[1], cs[it].dir[0]) for it in bac_it])

    bac_features_dict['Node_x1_x'].extend([cs[it].ends[0][0] for it in bac_it])
    bac_features_dict['Node_x1_y'].extend([cs[it].ends[0][1] for it in bac_it])
    bac_features_dict['Node_x2_x'].extend([cs[it].ends[1][0] for it in bac_it])
    bac_features_dict['Node_x2_y'].extend([cs[it].ends[1][1] for it in bac_it])

    # Surface area of a capsule:
    # S = 2πr(2r + a)
    bac_features_dict['AreaShape_Area'].extend(
        [2 * np.pi * cs[it].radius * (cs[it].length + 2 * cs[it].radius) for it in bac_it])

    bac_features_dict['TrackObjects_ParentImageNumber_50'].extend(this_time_step_bac_parent_image_number)
    bac_features_dict['TrackObjects_ParentObjectNumber_50'].extend(this_time_step_bac_parent_obj_number)
    bac_features_dict["TrackObjects_Label_50"].extend(this_time_step_bac_label)
    bac_features_dict['CellAge'].extend(this_time_step_bac_cell_age)
    bac_features_dict['id'].extend([cs[it].id for it in bac_it])

    # cell Type
    # In CellModeller CellTypes are: 0,1,2,3,...
    for index, cell_type_id in enumerate(cell_type_mapping.values()):
        bac_features_dict['Type'][index].extend(
            [int(cs[it].cellType == cell_type_id) for it in bac_it]
        )

    return bac_features_dict, bacteria_labels_dict, neighbor_records


def propagate_bacteria_labels(df, parent_image_number_col, parent_object_number_col, label_col):
    """
    Assign or propagate labels for bacteria based on parent-child relationships.

    This function assigns unique labels to bacteria and propagates them across frames based on
    parent-child relationships in tracking data. It ensures consistent labeling of bacteria
    across multiple time steps by leveraging parent image and object identifiers.

    :param pd.DataFrame df:
        Input dataframe containing bacterial tracking data, including parent-child relationships.
    :param str parent_image_number_col:
        Name of the column representing the parent image number.
    :param str parent_object_number_col:
        Name of the column representing the parent object number.
    :param str label_col:
        Name of the column to assign or propagate labels.

    :return:
        pd.DataFrame

        Updated dataframe with assigned or propagated labels in the specified column.
        Also adds a `checked` column to indicate whether a row has been processed.
    """

    df['checked'] = False
    df[label_col] = np.nan
    cond1 = df[parent_image_number_col] == 0

    df.loc[cond1, label_col] = df.loc[cond1, 'index'].values + 1
    df.loc[cond1, 'checked'] = True

    # other bacteria
    other_bac_df = df.loc[~ df['checked']]

    temp_df = df.copy()
    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    label_list = []

    same_bac_dict = {}

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if f'{int(parent_img_num)}_{int(parent_obj_num)}' not in same_bac_dict.keys():
            source_link = df.iloc[bac_index_dict[f'{int(parent_img_num)}_{int(parent_obj_num)}']]

            this_bac_label = source_link[label_col]

        else:

            this_bac_label = same_bac_dict[f'{int(parent_img_num)}_{int(parent_obj_num)}']

        label_list.append(this_bac_label)

        # same bacteria
        same_bac_dict[f'{int(image_number)}_{int(object_number)}'] = this_bac_label

    df.loc[other_bac_df.index, label_col] = label_list

    return df


def process_simulation_directory(input_directory, cell_type_mapping, output_directory, assign_cell_type=True,
                                 use_grandmother_as_parent=False, find_neighbors=True, pixel_per_micron=None,
                                 cellprofiler_orientation_format=False):
    """
    Processes a directory of CellModeller pickle files to extract cell features and track relationships.

    Parameters:
    - input_directory (str): Path to the directory containing CellModeller `.pickle` files.
    - cell_type_mapping (dict): Dictionary mapping cell type names to CellModeller IDs.
    - output_directory (str): Path to save output CSV files.
    - assign_cell_type (bool): If True, infer and assign cell types to tracked bacteria.
    - use_grandmother_as_parent (bool):
        If True, approximates the parent bacterium by selecting the nearest disappeared bacterium
        from the previous time step. Useful when large time step gaps cause the actual parent
        to no longer appear in the previous step.
    - find_neighbors (bool):
        If True, computes neighbor relationships between bacteria based on spatial proximity.
        Specifically, it defines two bacteria as neighbors if their expanded pixel boundaries
        touch — consistent with CellProfiler's "MeasureObjectNeighbors" module.
        Note: Even if this parameter is set to False, neighbor relationships will still be computed
        if the original pickle files do not already include neighbor data.
    pixel_per_micron (float or None):
        Spatial calibration factor converting microns → pixels.
        If provided, geometric quantities such as cell centers, endpoints, length, and
        radius are converted from microns to pixels. If None (default), no unit conversion is performed and values
        remain in raw simulation units.
    convert (bool):
        If True, converts CellModeller orientation angles into CellProfiler’s AreaShape_Orientation convention.

    Returns:
    - None. Writes two CSV files to the output directory:
        - 'Objects properties.csv' containing feature data.
        - 'Object relationships.csv' containing neighbor relationships.
    """

    # firstly I create a dictionary and append extracted features to corresponding key list
    dataframe = {'id': [], 'ImageNumber': [], 'ObjectNumber': [], 'Type': [], 'AreaShape_Area': [],
                 'AreaShape_Center_X': [], 'AreaShape_Center_Y': [], 'AreaShape_MajorAxisLength': [],
                 'AreaShape_MinorAxisLength': [], 'AreaShape_Orientation': [], 'Node_x1_x': [], 'Node_x1_y': [],
                 'Node_x2_x': [], 'Node_x2_y': [], 'CellAge': [], 'TrackObjects_ParentImageNumber_50': [],
                 'TrackObjects_ParentObjectNumber_50': [], 'validID': [], 'ImageName': [], 'TrackObjects_Label_50': []}

    rows_neighbors = []

    # keys: bacteria id
    # values: assigned bacteria labels
    bacteria_id_label = {}

    if assign_cell_type:
        dataframe['Type'] = [[] for _ in cell_type_mapping]

    # read pickle files
    path = input_directory + "/*.pickle"
    filename_list = [filename for filename in sorted(glob.glob(path))]

    for cnt, filename in enumerate(filename_list):

        # read current pickle file
        current_bacteria_info = pickle.load(open(filename, 'rb'))
        time_step = cnt + 1

        # read previous pickle file
        if cnt > 0:
            previous_bacteria = pickle.load(open(filename_list[cnt - 1], 'rb'))
        else:
            previous_bacteria = {'lineage': [], 'cellStates': []}

        # extract features
        dataframe, bacteria_id_label, rows_neighbors = \
            extract_bacterial_features(time_step, current_bacteria_info, previous_bacteria, dataframe,
                                       bacteria_id_label,
                                       rows_neighbors, cell_type_mapping, use_grandmother_as_parent)

    # create data frame
    df = pd.DataFrame({'ImageName': dataframe['ImageName'],
                       'ImageNumber': dataframe['ImageNumber'], 'ObjectNumber': dataframe['ObjectNumber'],
                       'AreaShape_Area': dataframe['AreaShape_Area'],
                       'AreaShape_Center_X': dataframe['AreaShape_Center_X'],
                       'AreaShape_Center_Y': dataframe['AreaShape_Center_Y'],
                       'AreaShape_MajorAxisLength': dataframe['AreaShape_MajorAxisLength'],
                       'AreaShape_MinorAxisLength': dataframe['AreaShape_MinorAxisLength'],
                       'AreaShape_Orientation': dataframe['AreaShape_Orientation'],
                       'Location_Center_X': dataframe['AreaShape_Center_X'],
                       'Location_Center_Y': dataframe['AreaShape_Center_Y'],
                       "TrackObjects_Label_50": dataframe["TrackObjects_Label_50"],
                       'TrackObjects_ParentImageNumber_50': dataframe['TrackObjects_ParentImageNumber_50'],
                       'TrackObjects_ParentObjectNumber_50': dataframe['TrackObjects_ParentObjectNumber_50'],
                       'id': dataframe['id'],
                       'CellAge': dataframe['CellAge'], 'Node_x1_x': dataframe['Node_x1_x'],
                       'Node_x1_y': dataframe['Node_x1_y'], 'Node_x2_x': dataframe['Node_x2_x'],
                       'Node_x2_y': dataframe['Node_x2_y']})

    df['LifeHistory'] = df.groupby('id')['id'].transform('size')

    if assign_cell_type:
        cell_type_names = cell_type_mapping.keys()
        for cnt, CellType in enumerate(cell_type_names):
            df[CellType] = dataframe['Type'][cnt]

        df['cellType'] = df[cell_type_names].idxmax(axis=1)
        df.loc[df[cell_type_names].max(axis=1) == 0, 'cellType'] = 0
        df.loc[(df[cell_type_names] == 1).sum(axis=1) > 1, 'cellType'] = 3

    # now check negative values
    x_axis_cols = ['Node_x1_x', 'Node_x2_x', 'Location_Center_X', 'AreaShape_Center_X']
    y_axis_cols = ['Node_x1_y', 'Node_x2_y', 'Location_Center_Y', 'AreaShape_Center_Y']

    x_min_val = np.min(df[x_axis_cols].to_numpy().flatten())
    y_min_val = np.min(df[y_axis_cols].to_numpy().flatten())

    if x_min_val < 0:
        df[x_axis_cols] += np.abs(x_min_val) + 1

    if y_min_val < 0:
        df[y_axis_cols] += np.abs(y_min_val) + 1

    neg_bac = df.loc[df['AreaShape_MajorAxisLength'] <= 0]
    id_for_update_age_list = []

    bac_id_should_remove_index = []
    for neg_bac_id in neg_bac['id'].unique():
        sel_bac = df.loc[df['id'] == neg_bac_id]
        if sel_bac['LifeHistory'].values[0] > 1:
            pos_len_bac = sel_bac.loc[sel_bac['AreaShape_MajorAxisLength'] > 0]
            neg_len_bac = sel_bac.loc[sel_bac['AreaShape_MajorAxisLength'] <= 0]

            pos_len_age = pos_len_bac['CellAge'].values
            neg_len_age = neg_len_bac['CellAge'].values

            if np.all(neg_len_age > pos_len_age[:, None]):
                bac_id_should_remove_index.extend(neg_len_bac.index.values.tolist())

                next_bac = df.loc[
                    (df['TrackObjects_ParentImageNumber_50'] == neg_len_bac['ImageNumber'].values[-1]) &
                    (df['TrackObjects_ParentObjectNumber_50'] == neg_len_bac['ObjectNumber'].values[-1])]

                if next_bac.shape[0] > 0:
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentImageNumber_50'] = 0
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentObjectNumber_50'] = 0

            elif np.all(neg_len_age < pos_len_age[:, None]):
                bac_id_should_remove_index.extend(neg_len_bac.index.values.tolist())
                id_for_update_age_list.append(neg_bac_id)

                next_bac = df.loc[
                    (df['TrackObjects_ParentImageNumber_50'] == neg_len_bac['ImageNumber'].values[-1]) &
                    (df['TrackObjects_ParentObjectNumber_50'] == neg_len_bac['ObjectNumber'].values[-1])]

                if next_bac.shape[0] > 0:
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentImageNumber_50'] = 0
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentObjectNumber_50'] = 0

            else:
                bac_id_should_remove_index.extend(pos_len_bac.index.values.tolist())
                bac_id_should_remove_index.extend(neg_len_bac.index.values.tolist())

                next_bac = df.loc[
                    (df['TrackObjects_ParentImageNumber_50'] == sel_bac['ImageNumber'].values[-1]) &
                    (df['TrackObjects_ParentObjectNumber_50'] == sel_bac['ObjectNumber'].values[-1])]

                if next_bac.shape[0] > 0:
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentImageNumber_50'] = 0
                    df.at[next_bac.index.values[0], 'TrackObjects_ParentObjectNumber_50'] = 0
        else:
            bac_id_should_remove_index.extend(sel_bac.index.values.tolist())

            next_bac = df.loc[
                (df['TrackObjects_ParentImageNumber_50'] == sel_bac['ImageNumber'].values[-1]) &
                (df['TrackObjects_ParentObjectNumber_50'] == sel_bac['ObjectNumber'].values[-1])]

            if next_bac.shape[0] > 0:
                df.at[next_bac.index.values[0], 'TrackObjects_ParentImageNumber_50'] = 0
                df.at[next_bac.index.values[0], 'TrackObjects_ParentObjectNumber_50'] = 0

    df = df.drop(bac_id_should_remove_index).reset_index(drop=True)

    mask_bac = df['id'].isin(id_for_update_age_list)
    df.loc[mask_bac, 'CellAge'] = df[mask_bac].groupby('id').cumcount() + 1

    df['LifeHistory'] = df.groupby('id')['id'].transform('size')

    df['index'] = df.index.values
    df = propagate_bacteria_labels(df, 'TrackObjects_ParentImageNumber_50',
                                   'TrackObjects_ParentObjectNumber_50',
                                   "TrackObjects_Label_50")
    df = df.drop('index', axis=1)
    df = df.drop('checked', axis=1)

    if pixel_per_micron is not None:
        # convert um to pixel
        df['AreaShape_Center_X'] /= pixel_per_micron
        df['AreaShape_Center_Y'] /= pixel_per_micron
        df['Location_Center_X'] /= pixel_per_micron
        df['Location_Center_Y'] /= pixel_per_micron

        df['AreaShape_MajorAxisLength'] /= pixel_per_micron
        df['AreaShape_MinorAxisLength'] /= pixel_per_micron

        df["Node_x1_x"] /= pixel_per_micron
        df["Node_x2_x"] /= pixel_per_micron
        df["Node_x1_y"] /= pixel_per_micron
        df["Node_x2_y"] /= pixel_per_micron

    if cellprofiler_orientation_format:
        # orientation
        df["AreaShape_Orientation"] = - (df["AreaShape_Orientation"] * 180 / np.pi) - 90

    # now we should check dataframes
    if len(rows_neighbors) == 0 or find_neighbors:
        # This means that the neighbors were not found in the CellModeler
        print('Finding Neighbors')
        df_neighbors = neighbor_finders(df)
    else:
        df_neighbors = pd.DataFrame(rows_neighbors, columns=['Image Number', 'First Object id',
                                                             'Second Object id'])

    # Convert 'Image Number' and 'First Object id' columns to string in both dataframes
    df_neighbors['Image Number'] = df_neighbors['Image Number'].astype(str)
    df_neighbors['First Object id'] = df_neighbors['First Object id'].astype(str)
    df_neighbors['Second Object id'] = df_neighbors['Second Object id'].astype(str)

    df['ImageNumber'] = df['ImageNumber'].astype(str)
    df['id'] = df['id'].astype(str)

    df_merge = df_neighbors.merge(df, left_on=['Image Number', 'First Object id'],
                                  right_on=['ImageNumber', 'id'], how='inner', suffixes=('_node', '_info'))

    df_final_merge = df_merge.merge(df, left_on=['Image Number', 'Second Object id'],
                                    right_on=['ImageNumber', 'id'], how='inner',
                                    suffixes=('_node_info', '_neighbor_info'))

    findl_df_neighbors = \
        df_final_merge[['Image Number', 'ObjectNumber_node_info', 'ObjectNumber_neighbor_info']].copy()

    findl_df_neighbors = findl_df_neighbors.rename({'Image Number': 'First Image Number',
                                                    'ObjectNumber_node_info': 'First Object Number',
                                                    'ObjectNumber_neighbor_info': 'Second Object Number'}, axis=1)

    findl_df_neighbors.insert(0, 'Relationship', 'Neighbors')
    findl_df_neighbors.insert(3, 'Second Image Number', findl_df_neighbors['First Image Number'].values)

    df['ImageNumber'] = df['ImageNumber'].astype(int)
    df['id'] = df['id'].astype(int)
    df = reassign_id(df, 'TrackObjects_ParentImageNumber_50',
                     'TrackObjects_ParentObjectNumber_50')
    df = df.drop('index', axis=1)

    # write to csv
    df.to_csv(output_directory + "/Objects_properties.csv", index=False)
    findl_df_neighbors.to_csv(output_directory + "/Object_relationships.csv", index=False)
