from __future__ import annotations
import pandas as pd
import copy
import warnings
from pathlib import Path
from importlib.resources import files
from typing import Optional, Dict, Any, Union, List, Sequence, Literal, Set, Iterable, Self
from .io import files as data_files
from .io import subset as data_subset
from .utils import data_utils as help_func
from .utils import phylo_utils as phylo_func

__all__ = [
    "MicrobiomeData"
]

class MicrobiomeData:
    """
    Container for microbiome data tables (abundance, taxonomy, metadata, sequences, tree).

    Attributes
    ----------
    tab : pd.DataFrame, optional
        Abundance table (features x samples).
    tax : pd.DataFrame, optional
        Taxonomy table (features x taxonomy levels).
    meta : pd.DataFrame, optional
        Metadata table (samples x variables).
    seq : pd.DataFrame, optional
        Sequence table (features x sequence).
    tree : pd.DataFrame, optional
        Phylogenetic tree.
    """

    def __init__(
        self,
        tab: Optional[pd.DataFrame] = None,
        tax: Optional[pd.DataFrame] = None,
        meta: Optional[pd.DataFrame] = None,
        seq: Optional[pd.DataFrame] = None,
        tree: Optional[pd.DataFrame] = None,
    ):
        self.tab = tab
        self.tax = tax
        self.meta = meta
        self.seq = seq
        self.tree = tree
        self._autocorrect()
        self._validate()

    #  ------------------------------------------------------------------------
    #  Methods for creating MicrobiomeData objects and importing data
    #  ------------------------------------------------------------------------
    @classmethod
    def load(cls, **kwargs) -> MicrobiomeData:
        """
        Load microbiome data from files and return a MicrobiomeData object.

        Parameters
        ----------
        kwargs : dict
            Arguments for file paths and parsing options, passed to the loader.

        Returns
        -------
        MicrobiomeData
            Loaded data object.

        Examples
        --------
        >>> data = MicrobiomeData.load(tab="otu_table.csv", meta="metadata.csv")
        """
        data = data_files.load(**kwargs)
        return cls(
            tab=data.get("tab"),
            tax=data.get("tax"),
            meta=data.get("meta"),
            seq=data.get("seq"),
            tree=data.get("tree"),
        )

    def add_tab(
        self,
        tab: str,
        *,
        path: str = "",
        sep: Optional[str] = None,
        taxonomy_levels: Optional[list[str]] = None
    ) -> Self:
        """
        Add or update `self.tab` (and `self.tax` if included in the file).

        Parameters
        ----------
        tab : str
            File name of the frequency table (.csv/.tsv, optionally gzipped, e.g. .csv.gz).
            Feature names (OTU/ASV/bin/MAG) should be in the first column (index).
        path : str, default ""
            Directory path (absolute or relative) containing `tab`. Can be "" for CWD.
        sep : str or None, default None
            Column separator. If None, pandas will attempt to auto-detect (engine='python').
        taxonomy_levels : list of str, optional
            Case-insensitive taxonomy column names to extract. Defaults to a broad set.

        Raises
        ------
        ValueError
            If the file cannot be read or has invalid format.

        Returns
        -------
        MicrobiomeData
            The updated object (self).

        """
        try:
            out = data_files.add_tab(
                tab,
                path=path,
                sep=sep,
                taxonomy_levels=taxonomy_levels,
            )
        except ValueError as e:
            # Add context and re-raise
            raise ValueError(f"[MicrobiomeData.add_tab] Failed to load '{tab}' from '{path}': {e}") from e

        # Assign results
        self.tab = out.get("tab")
        if "tax" in out:
            self.tax = out["tax"]
        self._autocorrect()
        self._validate()
        return self

    def add_tax(
        self,
        tax: str,
        *,
        path: str = "",
        sep: Optional[str] = None,
        add_taxon_prefix: bool = True,
    ) -> Self:
        """
        Add or update `self.tax`.

        Parameters
        ----------
        tax : str
            File name of the taxonomy table (.csv/.tsv, optionally gzipped, e.g. .csv.gz).
            Feature names (OTU/ASV/bin/MAG) should be in the first column (index).
        path : str, default ""
            Directory path (absolute or relative) containing `tab`. Can be "" for CWD.
        sep : str or None, default ","
            Column separator. If None, pandas will attempt to auto-detect (engine='python').
        add_taxon_prefix : bool, default True
            If True, add letters and two underscores before taxon names to indicate taxonomic level.

        Raises
        ------
        ValueError
            If the file cannot be read or has invalid format.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        try:
            out = data_files.add_tax(
                tax,
                path=path,
                sep=sep
            )
        except ValueError as e:
            raise ValueError(f"[MicrobiomeData.add_tax] Failed to load '{tax}' from '{path}': {e}") from e

        # Assign results
        self.tax = out.get("tax")
        self._autocorrect()
        self._validate()
        return self

    def add_seq_from_fasta(
        self,
        fasta: str,
        *,
        path: str = "",
        name_splitter: Optional[str] = None
    ) -> Self:
        """
        Add or update `self.seq`.

        Parameters
        ----------
        fasta : str
            Name of the FASTA file with sequences of OTUs or ASVs (.fa, .fasta, optionally gzipped).
        path : str, default ""
            Directory path (absolute or relative) containing `fasta`. Can be "" for CWD.
        name_splitter : str, optional
            If provided, splits sequence names on this delimiter and keeps the first part.

        Raises
        ------
        ValueError
            If `fasta` is missing or file cannot be read. If no sequences are found.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        try:
            out = data_files.add_seq_from_fasta(
                fasta,
                path=path,
                name_splitter=name_splitter
            )
        except ValueError as e:
            raise ValueError(f"[MicrobiomeData.add_seq_from_fasta] Failed to load '{fasta}' from '{path}': {e}") from e

        # Assign results
        self.seq = out.get("seq")
        self._autocorrect()
        self._validate()
        return self

    def add_tree(
        self,
        tree: str,
        *,
        path: str = ""
    ) -> Self:
        """
        Load tree from a newick file into a dictionary with a pandas DataFrame.
    
        Parameters
        ----------
        tree : str
            Name of the newick file with the tree.
        path : str, default ""
            Directory path (absolute or relative) containing `tree`. Can be "" for CWD.

        Raises
        ------
        ValueError
            If `tree` is missing or file cannot be read, or if no nodes are found.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        try:
            out = data_files.add_tree(
                tree,
                path=path,
            )
        except ValueError as e:
            raise ValueError(f"[MicrobiomeData.add_tree] Failed to load '{tree}' from '{path}': {e}") from e

        # Assign results
        self.tree = out.get("tree")
        self._autocorrect()
        self._validate()
        return self

    def add_meta(
        self,
        meta: str,
        *,
        path: str = "",
        sep: Optional[str] = ","
    ) -> Self:
        """
        Load meta data into a dictionary with a pandas DataFrame.
    
        Parameters
        ----------
        meta : str
            Name of the meta data file.
        path : str, default ""
            Directory path (absolute or relative) containing `meta`. Can be "" for CWD.
        sep : str or None, default ","
            Column separator. If None, pandas will attempt to auto-detect (engine='python').
    
        Raises
        ------
        ValueError
            If `meta` is missing or file cannot be read, or if no samples are found.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        try:
            out = data_files.add_meta(
                meta,
                path=path,
                sep=sep,
            )
        except ValueError as e:
            raise ValueError(f"[MicrobiomeData.add_meta] Failed to load '{meta}' from '{path}': {e}") from e

        # Assign results
        self.meta = out.get("meta")
        self._autocorrect()
        self._validate()
        return self

    def add_tax_from_sintax(
            self, 
            filename: str,
            *,
            path: str = "",
    ) -> Self:
        """
        Add or update taxonomy from a SINTAX output file.

        Parameters
        ----------
        filename : str
            Path to the SINTAX output file.
        path : str, default ""
            Directory path (absolute or relative) containing `sintax_file`. Can be "" for CWD.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        df = data_files.add_tax_from_sintax(filename=filename, path=path)
        self.tax = df # Update the tax attribute
        self._autocorrect()
        self._validate()
        return self

    def add_tax_from_qiime(
        self,
        filename: str,
        *,
        path: str = "",
    ) -> Self:
        """
        Add or update taxonomy from a QIIME2-style taxonomy file.

        Parameters
        ----------
        filename : str
            File name of the taxonomy table (.tsv, e.g. from QIIME2 export).
        path : str, default ""
            Directory path (absolute or relative) containing `tax`. Can be "" for CWD.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        df = data_files.add_tax_from_qiime(filename=filename, path=path)
        self.tax = df # Update the tax attribute
        self._autocorrect()
        self._validate()
        return self

    def add_tax_from_gtdbtk(
            self,
            filenames: Union[str, List[str]],
            *,
            path: str = "",
    ) -> Self:

        """
        Add or update taxonomy from one or more GTDB-Tk summary files.
        
        Parameters
        ----------
        filenames : str or list of str
            Path(s) to GTDB-Tk summary .tsv file(s).
        path : str, default ""
            Directory path (absolute or relative) containing file(s). Can be "" for CWD.

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        df = data_files.add_tax_from_gtdbtk(filenames=filenames, path=path)
        self.tax = df # Update the tax attribute
        self._autocorrect()
        self._validate()
        return self

    def add_tab_from_coverm(
        self,
        filename: str,
        *,
        path: str = "",
        first_sep: str = None,
        second_sep: str = " ",
        detection_threshold: float = None,
    ) -> Self:
        """
        Add a relative abundance table from a CoverM file.
    
        Parameters
        ----------
        filename : str
            Path to coverm .tsv or .csv file.
        path : str, default ""
            Directory path (absolute or relative) containing file(s). Can be "" for CWD.
        first_sep : str, optional
            Separator to help extract sample names from column headings.
        second_sep : str, optional
            Second separator to help extract sample names from column headings.
        detection_threshold : float, optional
            Detection threshold for relative abundance (default: None).

        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        result = data_files.add_tab_from_coverm(
            filename=filename,
            path=path,
            first_sep=first_sep,
            second_sep=second_sep,
            detection_threshold=detection_threshold
        )
        self.tab = result.get("tab")
    
        if "unmapped" in result:
            um_series = result["unmapped"]
            um_dataframe = pd.DataFrame(um_series.to_numpy(), index=um_series.index, columns=["unmapped_reads_perc"])
            if not hasattr(self, 'meta') or self.meta is None:
                self.meta = um_dataframe
            else:
                self.meta["unmapped_reads_perc"] = um_series
        self._autocorrect()
        self._validate()
        return self

    def add_ebd_tab_from_singlem(
        self,
        filename: str,
        *,
        path: str = "",
        first_sep: str = None,
        second_sep: str = " ",
    ) -> Self:
        """
        Add count data (tab) and taxonomic information (tax) from an EBD file generated
        with SingleM.
    
        Parameters
        ----------
        filename : str
            Path to the file (TSV).
        path : str, default ""
            Directory path (absolute or relative) containing file(s). Can be "" for CWD.
        first_sep : str or None
            Separator used to find sample name in CoverM-file column headings.
            Defaults to None.
        second_sep : str
            Separator used to find sample name in CoverM-file column headings.
            Defaults to None.
    
        Returns
        -------
        MicrobiomeData
            The updated object (self).
        """
        result = data_files.add_ebd_tab_from_singlem(
            filename=filename,
            path=path,
            first_sep=first_sep,
            second_sep=second_sep,
        )
        self.tab = result.get("tab")
        self.tax = result.get("tax")
        self._autocorrect()
        self._validate()
        return self

    #  ------------------------------------------------------------------------
    #  Methods saving files or showing information about the content
    #  ------------------------------------------------------------------------
    def save(self, path: str = '', savename: str = 'output', sep: str = ',') -> list:
        """
        Save frequency table, taxonomy, metadata, sequences, and tree to disk.

        Parameters
        ----------
        path : str, optional
            Directory path where files will be saved. Defaults to the current directory.
        savename : str, optional
            Base name for output files. Defaults to "output".
        sep : str, optional
            Field separator for CSV files. Defaults to ",".

        Returns
        -------
        list of str
            List of file paths that were saved.

        Examples
        --------
        >>> files = data.printout(path="results", savename="mydata")
        """
        return data_files.save(self.to_dict(), path=path, savename=savename, sep=sep)

    def info(self, preview_rows: int = 1)  -> None:
        """
        Print summary information about the MicrobiomeData object.

        Parameters
        ----------
        preview_rows : int, optional
            Number of rows to preview from metadata (default: 1).
        """
        print("MicrobiomeData object summary")
        print("-" * 40)
        # Abundance table
        if self.tab is not None:
            print(f"Abundance table: {self.tab.shape[0]} features x {self.tab.shape[1]} samples")
            print(f"Total reads: {self.tab.sum().sum()}")
            print(f"Min reads/sample: {self.tab.sum().min()}")
            print(f"Max reads/sample: {self.tab.sum().max()}")
        else:
            print("Abundance table: None")
        # Taxonomy
        if self.tax is not None:
            print(f"Taxonomy table: {self.tax.shape[0]} features, levels: {list(self.tax.columns)}")
        else:
            print("Taxonomy table: None")
        # Sequences
        if self.seq is not None:
            print(f"Sequence table: {self.seq.shape[0]} features")
        else:
            print("Sequence table: None")
        # Tree
        if self.tree is not None:
            print(f"Tree: {len(self.tree)} nodes")
        else:
            print("Tree: None")
        # Metadata
        if self.meta is not None:
            print(f"Metadata table: {self.meta.shape[0]} samples, columns: {list(self.meta.columns)}")
            if self.meta.shape[0] > 0:
                print("Metadata preview:")
                print(self.meta.head(preview_rows))
        else:
            print("Metadata table: None")
        print("-" * 40)

    def summarize_taxa(
        self,
        savename: str | None = None,
        *,
        path: str = "",
    ) -> pd.DataFrame:
        """
        Summarize the number of taxa at each taxonomic level per sample.
    
        Parameters
        ----------
        savename : str or None, default=None
            If provided, save the output table as CSV in the given path.
    
        Returns
        -------
        pandas.DataFrame
            Summary table with:

                - number of features per sample
                - total reads per sample
                - number of unique taxa at each taxonomic level
        """
    
        # --- Validate inputs ------------------------------------------------------
        if self.tax is None or not isinstance(self.tax, pd.DataFrame):
            raise ValueError("'tax' is missing or not a DataFrame.")
        if self.tab is None or not isinstance(self.tab, pd.DataFrame):
            raise ValueError("'tab' is missing or not a DataFrame.")
        if self.tax.shape[0] == 0 or self.tab.shape[0] == 0:
            raise ValueError("Features are missing in 'tax' or 'tab'.")
        if not self.tax.index.equals(self.tab.index):
            raise ValueError("Feature IDs in 'tax' and 'tab' must match.")
    
        tax = self.tax.copy()
        tab = self.tab.copy()
    
        taxlevels = tax.columns.tolist()
        samples = tab.columns.tolist()
        sample_sums = tab.sum(axis=0)
    
        # --- Build rows -----------------------------------------------------------
        rows = []
    
        for smp in ["Total"] + samples:
            row = {"Sample": smp}
    
            if smp == "Total":
                row["Features"] = tab.shape[0]
                row["Reads"] = sample_sums.sum()
                for tlev in taxlevels:
                    row[tlev] = tax[tlev].dropna().nunique()
            else:
                present = tab.index[tab[smp] > 0]
                row["Features"] = len(present)
                row["Reads"] = sample_sums[smp]
                for tlev in taxlevels:
                    row[tlev] = tax.loc[present, tlev].dropna().nunique()
    
            rows.append(row)
    
        output = pd.DataFrame(rows).set_index("Sample")
    
        if savename:
            file_path = Path(path) / f"{savename}.csv"
            output.to_csv(file_path)
    
        return output

    #  ------------------------------------------------------------------------
    #  Methods for subsetting and manipulating objects
    #  ------------------------------------------------------------------------
    def subset_samples(
        self,
        *,
        by: str = "index",
        values=None,
        exclude: bool = False,
        keep_absent: bool = False,
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Subset samples in the MicrobiomeData object using io.subset_samples.

        Parameters
        ----------
        by : str, default "index"
            How to select samples: "index" for sample names, or a column name in meta.
        values : list or scalar, optional
            Values to include (or exclude if exclude=True).
        exclude : bool, default False
            If True, exclude samples that match values.
        keep_absent : bool, default False
            If False, drop features (rows) with zero counts after subsetting.
        inplace : bool, default False
            If True, modify the object in place. If False, return a new object.

        Returns
        -------
        MicrobiomeData
            The filtered object (self if inplace=True, otherwise a new object).
        """
        return data_subset.subset_samples(
            self,
            by=by,
            values=values,
            exclude=exclude,
            keep_absent=keep_absent,
            inplace=inplace
        )

    def subset_features(
        self,
        *,
        featurelist=None,
        exclude: bool = False,
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Subset features (OTUs/ASVs/bins/MAGs) from a MicrobiomeData object 
        using io.subset_features. 

        Parameters
        ----------
        featurelist : list
            List of feature (OTU/ASV/bin) identifiers to keep or exclude.
        exclude : bool, default False
            If True, exclude values in featurelist instead of including them.
        inplace : bool, default False
            If True, mutate and return the same object. If False, return a new object.

        Returns
        -------
        MicrobiomeData
            The filtered object (self if inplace=True, otherwise a new object).
        """
        return data_subset.subset_features(
            self,
            featurelist=featurelist,
            exclude=exclude,
            inplace=inplace
        )

    def subset_abundant(
        self,
        *,
        n: int = 25,
        method: Literal["sum", "mean"] = "mean",
        exclude: bool = False,
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Subset features (OTUs/ASVs/bins/MAGs) from a MicrobiomeData object 
        using io.subset_abundant. 
    
        Parameters
        ----------
        n : int, default 25
            Number of top features to keep (or exclude if `exclude=True`).
            Values outside [0, n_features] are clamped to the valid range.
        method : {'sum','mean'}, default 'mean'
            Reduction across samples of **relative abundance** per feature.
            - 'sum'  : total relative abundance across samples
            - 'mean' : mean relative abundance across samples
        exclude : bool, default False
            If False (default), keep the top-n features.
            If True, exclude the top-n features (keep the rest).
        inplace : bool, default False
            Only relevant for MicrobiomeData input.
            If True, mutate the object and return it; otherwise, return a new object.
    
        Returns
        -------
        MicrobiomeData
            The filtered object (self if inplace=True, otherwise a new object).
        """
        return data_subset.subset_abundant(
            self,
            n=n,
            method=method,
            exclude=exclude,
            inplace=inplace
        )

    def merge_samples(
        self,
        *,
        by: Union[List[str], str],
        values: Optional[list] = None,
        method: str = "sum",
        keep_absent: bool = False,
        inplace: bool = False
    ) -> MicrobiomeData:
        """
        Merge samples in the MicrobiomeData object based on metadata grouping.
    
        Parameters
        ----------
        by : str or list
            Column(s) in metadata used for grouping samples.
        values : list, optional
            Metadata values to keep. If None, all unique values in `by` are used.
        method : {'sum', 'mean'}, default 'sum'
            Aggregation method for counts.
        keep_absent : bool, default False
            If False, remove features with zero counts after merging.
        inplace : bool, default False
            If True, modify this object in place; if False, return a new object.
    
        Returns
        -------
        MicrobiomeData
            Object with merged samples. Returns `self` if ``inplace=True``, otherwise a new
            `MicrobiomeData` instance.
    
        Raises
        ------
        ValueError
            If metadata or the specified column is missing, or if no samples match the specified values.
    
        Examples
        --------
        >>> data.merge_samples(by="Treatment", method="sum", inplace=True)
        >>> merged = data.merge_samples(by="Site", method="mean")
        """
        return data_subset.merge_samples(
            self,
            by=by,
            values=values,
            method=method,
            keep_absent=keep_absent,
            inplace=inplace
        )

    def subset_taxa(
        self,
        *,
        subset_levels: Optional[Union[str, Sequence[str]]] = None,
        subset_patterns: Optional[Union[str, Sequence[str]]] = None,
        exclude: bool = False,
        case: bool = False,
        regex: bool = True,
        match_type: Literal["contains", "fullmatch", "startswith", "endswith"] = "contains",
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Subset features (OTUs/ASVs/bins/MAGs) from the MicrobiomeData object based on taxonomic classification.
    
        Parameters
        ----------
        subset_levels : str or sequence of str, optional
            Taxonomic column(s) in which to search for patterns. If None, all columns in `tax` are used.
        subset_patterns : str or sequence of str
            Text patterns to identify taxa to keep. If a single string is passed, it is used as the only pattern.
        exclude : bool, default False
            If True, return taxa that do NOT match the given patterns (i.e., complement).
        case : bool, default False
            If True, pattern matching is case-sensitive.
        regex : bool, default True
            If True, patterns are treated as regex. If False, patterns are escaped (literal match).
        match_type : {'contains','fullmatch','startswith','endswith'}, default 'contains'
            Matching behavior applied to the strings in selected columns.
        inplace : bool, default False
            If True, mutate and return the same object. If False, return a new object.
    
        Returns
        -------
        MicrobiomeData
            Filtered object with updated 'tab', 'tax', and 'seq'. 'meta' and 'tree' are passed through.
    
        Raises
        ------
        ValueError
            If taxonomy table is missing or no patterns are provided.
    
        Examples
        --------
        >>> data.subset_taxa(subset_levels="Genus", subset_patterns="Bacteroides", inplace=True)
        >>> filtered = data.subset_taxa(subset_patterns=["Bacteroides", "Clostridium"], exclude=True)
        """
        return data_subset.subset_taxa(
            self,
            subset_levels=subset_levels,
            subset_patterns=subset_patterns,
            exclude=exclude,
            case=case,
            regex=regex,
            match_type=match_type,
            inplace=inplace
        )

    def rarefy(
        self,
        *,
        depth: Union[int, str] = "min",
        seed: Optional[int] = None,
        replacement: bool = False,
        inplace: bool = False
    ) -> MicrobiomeData:
        """
        Rarefy the abundance table to a fixed sequencing depth.
    
        This method is a thin wrapper around :func:`io.subset.rarefy`. It performs
        random subsampling (with or without replacement) to equalize sequencing depth
        across samples, then drops features and samples that become zero.
    
        Parameters
        ----------
        depth : int or 'min', default 'min'
            Target sequencing depth per sample. If 'min', the minimum depth across
            samples is used.
        seed : int, optional
            Random seed for reproducibility.
        replacement : bool, default False
            If True, sample with replacement (multinomial); otherwise sample
            without replacement.
        inplace : bool, default False
            If True, modify this object in place; if False, return a new object.
    
        Returns
        -------
        MicrobiomeData
            The rarefied object. Returns `self` if ``inplace=True``, otherwise a new
            `MicrobiomeData` instance.
    
        Notes
        -----
        - Rarefaction reduces sequencing depth variance across samples to facilitate
          certain diversity and dissimilarity analyses.
        - The exact algorithm and post‑processing (feature/sample pruning) are
          implemented in :func:`io.subset.rarefy`.
        - Index alignment and integrity are enforced via :meth:`_autocorrect` and
          :meth:`_validate` in the underlying implementation.
    
        Examples
        --------
        >>> data.rarefy(depth=10000, seed=42, inplace=True)
        >>> rarefied = data.rarefy(depth='min', replacement=True)
        """
        return data_subset.rarefy(
            self,
            depth=depth,
            seed=seed,
            replacement=replacement,
            inplace=inplace
        )

    def prune_tree(
        self,
        featurelist: Union[List[str], Set[str], Iterable[str], None] = None,
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Prune the tree to retain only branches whose leaves intersect with a given feature set,
        plus always keep the root branch.
    
        Parameters
        ----------
        featurelist : list of str or set of str or iterable of str, optional
            A collection of feature names to match against the leaves of each branch.
            If None, the method will attempt to use `self.tab.index.tolist()`.
        inplace : bool, default False
            If True, modify this object in place; if False, return a new object.

        Returns
        -------
        MicrobiomeData
            The object with pruned tree. Returns `self` if ``inplace=True``, otherwise a new
            `MicrobiomeData` instance.
        """
        if self.tree is None:
            raise ValueError("'tree' is missing")
    
        if featurelist is None and self.tab is not None:
            featurelist = self.tab.index.tolist()
    
        if featurelist is None or len(featurelist) == 0:
            raise ValueError("Either 'featurelist' must be provided or 'self.tab' must have a valid index.")

        if inplace:
            self.tree = phylo_func.subset_tree(self.tree, featurelist)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.tree = phylo_func.subset_tree(new_obj.tree, featurelist)
            return new_obj

    def rename_features(
        self,
        name_type: str = 'OTU',
        inplace: bool = False,
    ) -> MicrobiomeData:
        """
        Rename feature identifiers (row indices) based on their relative abundance or taxonomic order.
    
        The renaming assigns new feature names in the format `{name_type}{i}`, where `i`
        is the rank of the feature after sorting:
        - By mean relative abundance if `tab` (abundance table) is present.
        - By taxonomic order if `tax` is present and `tab` is absent.
    
        Parameters
        ----------
        name_type : str, default='OTU'
            Prefix for new feature names (e.g., 'OTU', 'ASV').
        inplace : bool, default=False
            If True, modify object in place.

        Returns
        -------
        MicrobiomeData
            The updated object. If `inplace=True`, returns self; otherwise, a new instance.
        """

        return help_func.rename_features(
            self,
            name_type=name_type,
            inplace=inplace,
        )

    def tax_prefix(
        self,
        add: bool = True,
        inplace: bool = False,
        custom_prefix: Dict[str, str] = None
    ) -> MicrobiomeData:
        """
        Add or remove prefix (e.g. d__, p__) to taxonomic classifications.
    
        Parameters
        ----------
        add : bool, default=True
            If True, add prefix. If False, remove prefix.
        inplace : bool, default=False
            If True, modify object in place.
        custom_prefix : dict, default=None
            A dictionary with taxonomic levels as keys and prefix as values.

        Returns
        -------
        MicrobiomeData
            The updated object. If `inplace=True`, returns self; otherwise, a new instance.
        """
        return help_func.tax_prefix(
            self,
            add=add,
            inplace=inplace,
            custom_prefix=custom_prefix
        )

    
    #  ------------------------------------------------------------------------
    #  Utility methods
    #  ------------------------------------------------------------------------
    @classmethod
    def load_example(
        cls, 
        example_name: str = "Modin_et_al_2025",
    ) -> MicrobiomeData:
        """
        Load a MicrobiomeData object from packaged example files.

        Parameters
        ----------
        example_name : str
            Name of the example to load. Options:

            - "Modin2025": Uses CoverM and GTDB-Tk output files from the Modin et al. study https://doi.org/10.1111/1751-7915.70238.
            - "Saheb-Alam2019_DADA2": Uses qiime2-dada2 output files from the Saheb-Alam et al. study https://doi.org/10.1111/1751-7915.13449.
            - "Saheb-Alam2019_Deblur": Uses qiime2-deblur output files from the Saheb-Alam et al. study https://doi.org/10.1111/1751-7915.13449.

        Returns
        -------
        MicrobiomeData
            An instance loaded with example data.

        Raises
        ------
        ValueError
            If the example_name is not recognized.
        """
        if example_name == "Modin2025" or "modin" in example_name.lower():
            # Example: CoverM and GTDB-Tk files
            base = "qdiv.example_data"
            tab_file = files(base).joinpath("Modin2025_CoverM.tsv")
            tax_file1 = files(base).joinpath("Modin2025_gtdbtk.ar53.summary.tsv")
            tax_file2 = files(base).joinpath("Modin2025_gtdbtk.bac120.summary.tsv")
            meta_file = files(base).joinpath("Modin2025_metadata.csv")
            tree_file = files(base).joinpath("Modin2025_tree.nwk")

            obj = cls()
            obj.add_tab_from_coverm(str(tab_file), detection_threshold=0.5, 
                                    first_sep="/", second_sep="_R")
            obj.add_tax_from_gtdbtk([str(tax_file1), str(tax_file2)])
            obj.add_meta(str(meta_file))
            obj.add_tree(str(tree_file))
            return obj

        elif example_name == "Saheb-Alam2019_DADA2" or ("alam" in example_name.lower() and "dada2" in example_name.lower()):
            # Example: CoverM and GTDB-Tk files
            base = "qdiv.example_data"
            tab_file = files(base).joinpath("Saheb-Alam2019_tab_dada2.tsv")
            tax_file = files(base).joinpath("Saheb-Alam2019_tax_dada2.tsv")
            seq_file = files(base).joinpath("Saheb-Alam2019_seq_dada2.fasta")
            meta_file = files(base).joinpath("Saheb-Alam2019_meta.csv")
            tree_file = files(base).joinpath("Saheb-Alam2019_tree_dada2.nwk")

            obj = cls()
            obj.add_tab(str(tab_file))
            obj.add_tax_from_qiime(str(tax_file))
            obj.add_seq_from_fasta(str(seq_file))
            obj.add_meta(str(meta_file))
            obj.add_tree(str(tree_file))
            return obj

        elif example_name == "Saheb-Alam2019_Deblur" or ("alam" in example_name.lower() and "deblur" in example_name.lower()):
            # Example: CoverM and GTDB-Tk files
            base = "qdiv.example_data"
            tab_file = files(base).joinpath("Saheb-Alam2019_tab_deblur.tsv")
            tax_file = files(base).joinpath("Saheb-Alam2019_tax_deblur.tsv")
            seq_file = files(base).joinpath("Saheb-Alam2019_seq_deblur.fasta")
            meta_file = files(base).joinpath("Saheb-Alam2019_meta.csv")
            tree_file = files(base).joinpath("Saheb-Alam2019_tree_deblur.nwk")

            obj = cls()
            obj.add_tab(str(tab_file))
            obj.add_tax_from_qiime(str(tax_file))
            obj.add_seq_from_fasta(str(seq_file))
            obj.add_meta(str(meta_file))
            obj.add_tree(str(tree_file))
            return obj

        elif example_name is None:
            obj = cls()
            return obj
            
        else:
            raise ValueError(
                f"Unknown example_name '{example_name}'. "
                "Available options: 'Modin2025', 'Saheb-Alam2019_DADA2', 'Saheb-Alam2019_Deblur'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the data as a dictionary.

        Returns
        -------
        dict
            Dictionary with keys: 'tab', 'tax', 'meta', 'seq', 'tree'.
        """
        return {
            "tab": self.tab,
            "tax": self.tax,
            "meta": self.meta,
            "seq": self.seq,
            "tree": self.tree,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MicrobiomeData:
        """
        Create a MicrobiomeData object from a dictionary.
    
        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'tab' : pd.DataFrame (required)
            - 'tax' : pd.DataFrame, optional
            - 'meta' : pd.DataFrame, optional
            - 'seq' : pd.DataFrame, optional
            - 'tree' : pd.DataFrame, optional
    
        Returns
        -------
        MicrobiomeData
            A new MicrobiomeData object initialized from the dictionary.
    
        Raises
        ------
        ValueError
            If 'tab' is missing or not a pandas DataFrame.
    
        Examples
        --------
        >>> my_dict = {
        ...     "tab": pd.DataFrame(...),
        ...     "tax": pd.DataFrame(...),
        ...     "meta": pd.DataFrame(...)
        ... }
        >>> obj = MicrobiomeData.from_dict(my_dict)
        """
        if "tab" not in data or not isinstance(data["tab"], pd.DataFrame):
            raise ValueError("Input dictionary must contain a 'tab' key with a pandas DataFrame.")
    
        return cls(
            tab=data.get("tab"),
            tax=data.get("tax"),
            meta=data.get("meta"),
            seq=data.get("seq"),
            tree=data.get("tree"),
        )

    def _autocorrect(self):
        """
        Automatically align indices between tables and warn the user if corrections are made.
        """
        # Sort index and fix index column name
        if self.tab is not None and len(self.tab) > 0:
            self.tab = help_func.sort_index_by_number(self.tab)
            self.tab.index.name = 'Feature'
        if self.seq is not None and len(self.seq) > 0:
            self.seq = help_func.sort_index_by_number(self.seq)
            self.seq.index.name = 'Feature'
        if self.tax is not None and len(self.tax) > 0:
            self.tax = help_func.sort_index_by_number(self.tax)
            self.tax.index.name = 'Feature'
        
        if self.tab is not None and self.tax is not None:
            common_features = self.tab.index.intersection(self.tax.index)
            dropped_tab = set(self.tab.index) - set(common_features)
            dropped_tax = set(self.tax.index) - set(common_features)
            if dropped_tab or dropped_tax:
                msg = (
                    f"Auto-correct: Subsetting 'tab' and 'tax' to {len(common_features)} common features.\n"
                    f"  Dropped {len(dropped_tab)} from 'tab': {list(dropped_tab)[:5]}{'...' if len(dropped_tab) > 5 else ''}\n"
                    f"  Dropped {len(dropped_tax)} from 'tax': {list(dropped_tax)[:5]}{'...' if len(dropped_tax) > 5 else ''}\n"
                    "  (Check your input files for consistent feature names.)"
                )
                warnings.warn(msg, UserWarning)
                self.tab = self.tab.loc[common_features]
                self.tax = self.tax.loc[common_features]

        # Align features (rows) between tab and seq
        if self.tab is not None and self.seq is not None:
            common_features = self.tab.index.intersection(self.seq.index)
            dropped_tab = set(self.tab.index) - set(common_features)
            dropped_seq = set(self.seq.index) - set(common_features)
            if dropped_tab or dropped_seq:
                msg = (
                    f"Auto-correct: Subsetting 'tab' and 'seq' to {len(common_features)} common features.\n"
                    f"  Dropped {len(dropped_tab)} from 'tab': {list(dropped_tab)[:5]}{'...' if len(dropped_tab) > 5 else ''}\n"
                    f"  Dropped {len(dropped_seq)} from 'seq': {list(dropped_seq)[:5]}{'...' if len(dropped_seq) > 5 else ''}\n"
                    "  (Check your input files for consistent feature names.)"
                )
                warnings.warn(msg, UserWarning)
                self.tab = self.tab.loc[common_features]
                self.seq = self.seq.loc[common_features]
                self.tab.index.name = 'Features'
                self.seq.index.name = 'Features'

        # Align samples (columns) between tab and meta
        if self.tab is not None and self.meta is not None:
            common_samples = self.tab.columns.intersection(self.meta.index)
            dropped_tab = set(self.tab.columns) - set(common_samples)
            dropped_meta = set(self.meta.index) - set(common_samples)
            if dropped_tab or dropped_meta:
                msg = (
                    f"Auto-correct: Subsetting 'tab' columns and 'meta' index to {len(common_samples)} common samples.\n"
                    f"  Dropped {len(dropped_tab)} from 'tab': {list(dropped_tab)[:5]}{'...' if len(dropped_tab) > 5 else ''}\n"
                    f"  Dropped {len(dropped_meta)} from 'meta': {list(dropped_meta)[:5]}{'...' if len(dropped_meta) > 5 else ''}\n"
                    "  (Check your input files for consistent sample names.)"
                )
                warnings.warn(msg, UserWarning)
                self.tab = self.tab[common_samples]
                self.meta = self.meta.loc[common_samples]
            self.tab = self.tab[self.meta.index]


    def _validate(self):
        """
        Internal validation to ensure index alignment, uniqueness, and data integrity.
        Raises ValueError if inconsistencies or duplicates are found.
        """
        if self.tab is not None:
            # Check for empty tab
            if len(self.tab) == 0:
                raise ValueError("Features missing in tab.")
            if len(self.tab.columns) == 0:
                raise ValueError("Samples missing in tab.")
    
            # Check for duplicate feature names (rows) in tab
            if self.tab.index.has_duplicates:
                dups = self.tab.index[self.tab.index.duplicated()].unique().tolist()
                raise ValueError(f"Duplicate feature names in tab: {dups}")
    
            # Check for duplicate sample names (columns) in tab
            if self.tab.columns.has_duplicates:
                dups = self.tab.columns[self.tab.columns.duplicated()].unique().tolist()
                raise ValueError(f"Duplicate sample names in tab: {dups}")
    
            # Taxonomy checks
            if self.tax is not None:
                if len(self.tax) == 0:
                    raise ValueError("Features missing in tax.")
                if self.tax.index.has_duplicates:
                    dups = self.tax.index[self.tax.index.duplicated()].unique().tolist()
                    raise ValueError(f"Duplicate feature names in tax: {dups}")
                if not self.tab.index.equals(self.tax.index):
                    raise ValueError("Indices of 'tab' and 'tax' do not match.")
    
            # Sequence checks
            if self.seq is not None:
                if len(self.seq) == 0:
                    raise ValueError("Features missing in seq.")
                if self.seq.index.has_duplicates:
                    dups = self.seq.index[self.seq.index.duplicated()].unique().tolist()
                    raise ValueError(f"Duplicate feature names in seq: {dups}")
                if not self.tab.index.equals(self.seq.index):
                    raise ValueError("Indices of 'tab' and 'seq' do not match.")
    
            # Metadata checks
            if self.meta is not None:
                if len(self.meta) == 0:
                    raise ValueError("Samples missing in meta.")
                if self.meta.index.has_duplicates:
                    dups = self.meta.index[self.meta.index.duplicated()].unique().tolist()
                    raise ValueError(f"Duplicate sample names in meta: {dups}")
                if not self.tab.columns.equals(self.meta.index):
                    raise ValueError("Sample names in 'tab' and 'meta' do not match.")
    
            # Tree checks
            if self.tree is not None:
                tab_features = set(self.tab.index)
                tree_nodes = set(self.tree['nodes'])
                if not tab_features.issubset(tree_nodes):
                    raise ValueError("Not all tab features are found among tree nodes.")

        if self.tax is not None:
            if len(self.tax) == 0:
                raise ValueError("Features missing in tax.")
            if len(self.tax.columns) == 0:
                raise ValueError("Tax levels in tab.")

        if self.meta is not None:
            if len(self.meta) == 0:
                raise ValueError("Samples missing in meta.")

        if self.seq is not None:
            if len(self.seq) == 0:
                raise ValueError("Features missing in seq.")

        if self.tree is not None:
            if len(self.tree) == 0:
                raise ValueError("Features missing in tree.")

    def __repr__(self):
        n_features = self.tab.shape[0] if self.tab is not None else 0
        n_samples = self.tab.shape[1] if self.tab is not None else 0
        return (f"<MicrobiomeData: {n_features} features, {n_samples} samples, "
                f"tax={'yes' if self.tax is not None else 'no'}, "
                f"meta={'yes' if self.meta is not None else 'no'}, "
                f"seq={'yes' if self.seq is not None else 'no'}, "
                f"tree={'yes' if self.tree is not None else 'no'}>")
