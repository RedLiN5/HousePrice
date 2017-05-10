import matplotlib.pyplot as plt

def RelationPlot(data = None,
                 x = None,
                 y = None,
                 is_discrete = None,
                 output_dir = None):
    """
    Plot relation plot between two features in a pandas.DataFrame

    Parameters
    ----------
    :param data: pandas.DataFrame
    :param x: String, column name
    :param y: String, column name
    :param is_discrete: boolean
        If True, both of x and y are discrete variables.
        If False, at least one of x and y is a continuous variable.
    :param output_dir: String, plot output directory
        Default: In current working directory.
    """
    if any(item is None for item in [data, x, y]):
        raise ValueError("Any item in 'data', 'x' and 'y' cannot be None")
    NA_index = data[y].isnull()
    if is_discrete:
        df = data.ix[~NA_index, [x, y]].groupby([x, y]).size().reset_index(name="Time")
        for i in range(df.shape[0]):
            plt.scatter(df.ix[i, x],
                        df.ix[i, y],
                        s=df.ix[i, 'Time'],
                        c='b')
    else:
        plt.scatter(data.ix[~NA_index, x],
                    data.ix[~NA_index, y],
                    c='b')
    plt.xlabel(x)
    plt.ylabel(y)
    if output_dir is None:
        plt.savefig('{0}_{1}.png'.format(x, y))
    else:
        plt.savefig('{0}/{1}_{2}.png'.format(output_dir, x, y))


