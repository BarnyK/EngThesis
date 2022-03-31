from data.indexes import index_set


def test_indexes(args: dict):
    try:
        train, test = index_set(**args)
        print(f"Dataset {args.get('name')!r} total images: {len(train)+len(test)} training: {len(train)}, test: {len(test)}")
        print("Test succeeded")
    except FileNotFoundError as er:
        print(er)
        print("The path specified is incorrect or the data inside is not as expected")
    except ValueError as er:
        print(er)
    except KeyError as er:
        print(er)
    except TypeError as er:
        print(er)
