class Abstract_Preprocess:
    def __init__(self):
        pass

    def _format_df(self, df, feature_types):
        raise NotImplementedError()

    def _remove_object(self, df, feature_types):
        raise NotImplementedError()

    def _numerize_category(self, df_col):
        raise NotImplementedError()

    def auto_preprocess(cls, data_raw, key_types, feature_types, dropna=False):
        pass
