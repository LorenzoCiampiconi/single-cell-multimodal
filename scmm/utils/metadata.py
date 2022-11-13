def map_to_dataset(meta_row):
    tech = meta_row["technology"]
    day = meta_row["day"]
    donor = meta_row["donor"]
    match (tech, day, donor):
        case "citeseq", 7, _:
            return "private"
        case "citeseq", _, 27678:
            return "public"
        case "citeseq", _, _:
            return "train"
        case "multiome", 10, _:
            return "private"
        case "multiome", _, 27678:
            return "public"
        case "multiome", _, _:
            return "train"

    raise ValueError
