"""
Data Relabeling/Aggregation Transformation (Eco/BioCyc)
"""
from typing import override

import pandas as pd
import polars as pl
import numpy as np

from ecoli.library.transform.data_transformer import DataTransformer, downsample


class DataTransformerBioCyc(DataTransformer):
    @override
    def _transform(
            self,
            experiment_id: str,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        df = None
        transform_type = kwargs.get("type", "genes")
        if transform_type == "genes":
            df = self._genes(outputs_loaded, observable_ids, lazy)
        elif transform_type == "bulk":
            df = self._bulk(outputs_loaded, observable_ids, lazy)

        elif transform_type == "reactions":
            df = self._reactions(outputs_loaded, observable_ids, lazy)
        if df is None:
            raise ValueError("You must specify a type of transformation")
        return df

    def _genes(
            self,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        mrna_select = self.data_labels.mrna_cistron_names

        mrna_mtx = np.stack(outputs_loaded["listeners__rna_counts__full_mRNA_cistron_counts"])

        mrna_idxs = [self.data_labels.mrna_cistron_names.index(gene_id) for gene_id in mrna_select]

        mrna_trajs = [mrna_mtx[:, mrna_idx] for mrna_idx in mrna_idxs]

        mrna_plot_dict = {key: val for (key, val) in zip(mrna_select, mrna_trajs)}

        mrna_plot_dict["time"] = outputs_loaded["time"]

        # mrna_plot_df = pd.DataFrame(mrna_plot_dict)
        # mrna_df_long = mrna_plot_df.melt(
        #     id_vars=["time"],  # Columns to keep as identifier variables
        #     var_name="gene names",  # Name for the new column containing original column headers
        #     value_name="counts",  # Name for the new column containing original column values
        # )
        mrna_df_long = pl.LazyFrame(
            pd.DataFrame(mrna_plot_dict).melt(
                id_vars=["time"],  # Columns to keep as identifier variables
                var_name="gene names",  # Name for the new column containing original column headers
                value_name="counts",  # Name for the new column containing original column values
            )
        )

        # mrna_df = DataTransformer._downsample_dataframe(mrna_df_long)
        mrna_df: pl.LazyFrame = downsample(mrna_df_long)

        # return mrna_df[mrna_df["gene names"].isin(observable_ids)] if observable_ids is not None else mrna_df
        genes_data: pl.LazyFrame = mrna_df.filter(
            pl.col("gene names").is_in(observable_ids)
        )
        return genes_data if lazy else genes_data.collect()

    def _bulk(
            self,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        pass

    def _reactions(
            self,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        pass


class DataTransformerGenes(DataTransformer):
    @override
    def _transform(
        self,
        experiment_id: str,
        outputs_loaded: pd.DataFrame,
        observable_ids: list[str] | None = None,
        lazy: bool = True,
        **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        mrna_select = self.data_labels.mrna_cistron_names

        mrna_mtx = np.stack(outputs_loaded["listeners__rna_counts__full_mRNA_cistron_counts"])

        mrna_idxs = [self.data_labels.mrna_cistron_names.index(gene_id) for gene_id in mrna_select]

        mrna_trajs = [mrna_mtx[:, mrna_idx] for mrna_idx in mrna_idxs]

        mrna_plot_dict = {key: val for (key, val) in zip(mrna_select, mrna_trajs)}

        mrna_plot_dict["time"] = outputs_loaded["time"]

        # mrna_plot_df = pd.DataFrame(mrna_plot_dict)
        # mrna_df_long = mrna_plot_df.melt(
        #     id_vars=["time"],  # Columns to keep as identifier variables
        #     var_name="gene names",  # Name for the new column containing original column headers
        #     value_name="counts",  # Name for the new column containing original column values
        # )
        mrna_df_long = pl.LazyFrame(
            pd.DataFrame(mrna_plot_dict).melt(
                id_vars=["time"],  # Columns to keep as identifier variables
                var_name="gene names",  # Name for the new column containing original column headers
                value_name="counts",  # Name for the new column containing original column values
            )
        )

        # mrna_df = DataTransformer._downsample_dataframe(mrna_df_long)
        mrna_df: pl.LazyFrame = downsample(mrna_df_long)

        # return mrna_df[mrna_df["gene names"].isin(observable_ids)] if observable_ids is not None else mrna_df
        genes_data: pl.LazyFrame = mrna_df.filter(
            pl.col("gene names").is_in(observable_ids)
        )
        return genes_data if lazy else genes_data.collect()


class DataTransformerBulk(DataTransformer):
    @override
    def _transform(
            self,
            experiment_id: str,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        pass


class DataTransformerReactions(DataTransformer):
    @override
    def _transform(
            self,
            experiment_id: str,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        pass
