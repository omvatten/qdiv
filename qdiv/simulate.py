import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Returns a dataframe with simulated communities with log-normal species abundance distribution.
# size is the number of species in the community
# communities is the number of communities in the output dataframe
# sigma is the standard deviation of the sampled distribution. A low value will generate a community where all species have similar relative abundance and vice versa.
# c_prefix is the prefix for the numbered communities in the output dataframe.
# species_prefix is the prefix for the numbered species in the output dataframe
def community(size=100, communities=1, sigma=1, c_prefix='Comm', species_prefix='OTU'):
    clist = np.arange(communities)+1
    clist = clist.astype(str)
    clist = np.char.add(c_prefix, clist)

    ixlist = np.arange(size)+1
    ixlist = ixlist.astype(str)
    ixlist = np.char.add(species_prefix, ixlist)

    df = pd.DataFrame(index=ixlist, columns=clist)

    for c in clist:
        ralist = np.random.lognormal(mean=0, sigma=sigma, size=size)
        ralist = np.sort(ralist)[::-1]
        df[c] = ralist

    df = 100*df/df.sum()
    return df

# Returns a dataframe with a sample community given a species abundance distribution.
# community is a dataframe with the species abundance distribution of a community. It can be genered with the simulate.community function.
# n is the sample size (i.e. the sampled individuals)
def sample(community, n=10000):
    if isinstance(community, pd.DataFrame):
        smp_out = pd.DataFrame(0, index=community.index, columns=community.columns)
        for c in community.columns:
            rand_sample = community[[c]].sample(n=100000, replace=True, weights=community[c])
            rand_sample['OTU'] = rand_sample.index
            rand_sample = rand_sample.groupby('OTU').count()
            smp_out.loc[rand_sample.index, c] = rand_sample[c]
        return smp_out
    else:
        return None

# Returns a dataframe with a species abundance distribution for assembled communities.
# community is a dataframe with the species abundance distribution of a community. It can be genered with the simulate.community function.
# immigrants is a dataframe with the species abundance distribution of an immigrating community
# fitness is a dataframe with fitness of the taxa in the assembled community
# selection and dispersal give the relative importance of the two processes
def assembly(community, immigrants, fitness, selection=1, dispersal=1):
    if not isinstance(community, pd.DataFrame) or not isinstance(immigrants, pd.DataFrame) or not isinstance(fitness, pd.DataFrame):
        print('Error: input dataframes missing.')
        return None

    c_out = community.copy()
    change = pd.DataFrame(1, index=c_out.index, columns=c_out.columns)
    mech = np.array([selection, dispersal])
    mech = mech/np.sum(mech)
    counter = 0
    while sum(change.sum())>0.001 or counter < 100:
        counter += 1
        old_c_out = c_out.copy()                   
        for c in c_out.columns:
            frac_fitness = c_out[c]*0.1*mech[0]*fitness[c]
            frac_immigrants = c_out[c]*0.1*mech[1]*immigrants[c]
            c_out[c] = frac_fitness + frac_immigrants
            c_out[c] = 100*c_out[c]/c_out[c].sum()
        change = c_out - old_c_out
        change = change.pow(2)
        print(sum(change.sum()))
    return c_out
        
