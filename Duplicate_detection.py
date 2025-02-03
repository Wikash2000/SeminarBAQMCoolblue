import pandas as pd


def filter_grp_groups(file_path, grp_range=None, duplicate_filters=None):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Define bins for grouping
    bins = [0, 1, 3, 5, 10, 20, float('inf')]
    labels = ["[0,1)", "[1,3)", "[3,5)", "[5,10)", "[10,20)", "[20,∞)"]

    # Create a new column categorizing GRP into bins
    df['GRP_group'] = pd.cut(df['indexed_gross_rating_point'], bins=bins, labels=labels, right=False)

    # Apply GRP range filter if specified
    if grp_range:
        df = df[df['GRP_group'] == grp_range]

    # Apply duplicate filter if specified
    if duplicate_filters:
        df = df[df['Duplicates'].isin(duplicate_filters)]

    # Calculate GRP percentage difference for duplicates
    grouped = df.groupby(df['Duplicates'].eq('Duplicate first').cumsum())

    grp_percentage_list = []
    high_percentage_count = 0  # Counter for groups where one value is at least 70% of total
    low_gap_count = 0  # Counter for groups where remaining values are all below 2 and not in the 70% group
    total_groups = len(grouped)  # Count total groups
    high_percentage_groups = set()  # Track groups with a high percentage value
    low_gap_groups = []  # List of groups where all values are below 2
    remaining_groups = []  # List of remaining groups

    for group_id, group in grouped:
        if 'Duplicate first' in group['Duplicates'].values:
            total_grp = group['indexed_gross_rating_point'].sum()
            percentage_values = (group['indexed_gross_rating_point'] / total_grp) * 100
            grp_percentage_list.extend(percentage_values)

            # Check if any percentage is at least 70%
            if any(percentage_values >= 70):
                high_percentage_count += 1
                high_percentage_groups.add(group_id)
        else:
            grp_percentage_list.extend([None] * len(group))

    # Check for groups where all values are below 2 and not in high_percentage_groups
    for group_id, group in grouped:
        if group_id not in high_percentage_groups:
            if all(group['indexed_gross_rating_point'] < 2):
                low_gap_count += 1
                low_gap_groups.append(
                    group[["date", "time", "Duplicates", "channel", "indexed_gross_rating_point"]])
            else:
                remaining_groups.append(
                    group[["date", "time", "Duplicates", "channel", "indexed_gross_rating_point"]])

    df.insert(df.columns.get_loc("indexed_gross_rating_point") + 1, "GRP_percentage", grp_percentage_list)

    return df, high_percentage_count, total_groups, low_gap_count, low_gap_groups, remaining_groups


# Example usage
file_path = "Broadcasting_data_duplicates_bothGRPhigh.xlsx"
grp_range = None #"[5,10)"  # Change this to filter a different range
duplicate_filters = ["Duplicate first",
                     "Duplicate"]  # Choose one or more options: "Duplicate first", "Duplicate", "False"

df_selected, high_percentage_count, total_groups, low_gap_count, low_gap_groups, remaining_groups = filter_grp_groups(file_path, grp_range, duplicate_filters)

# Display the full filtered dataframe
print(df_selected.to_string(index=False))

# Display the count of commercials in the selected GRP range or the whole dataset
if grp_range:
    print(f"Total commercials in GRP range {grp_range} with filters {duplicate_filters}: {len(df_selected)}")
else:
    print(f"Total commercials in the entire dataset with filters {duplicate_filters}: {len(df_selected)}")

# Display the count of groups and how many have at least one value >= 70%
print(f"Total groups: {total_groups}")
print(f"Group 1: total groups where at least one value is >= 80% of total GRP: {high_percentage_count}")
print(f"Group 2: total groups where all values are below 2 and not in the 80% group: {low_gap_count}")
print(f"Sum of group 1 and 2: {high_percentage_count+low_gap_count}")

# Display the list of groups where all values are below 2
#print("Groups where all values are below 2:")
#for group in low_gap_groups:
#    print(group.to_string(index=False))

# Display the list of remaining groups
print("Remaining groups:")
for group in remaining_groups:
    print(group.to_string(index=False))