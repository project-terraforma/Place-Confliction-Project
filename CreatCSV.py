import pandas as pd
import json

def first_item(x):
    # 空值
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    # 已经是 Python list/tuple
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else ""
    # 可能是 PyArrow 的 ListValue
    try:
        import pyarrow as pa
        if isinstance(x, pa.lib.ListValue):
            return x[0].as_py() if len(x) > 0 else ""
    except Exception:
        pass
    # 字符串：尝试按 JSON 解析（既支持 ["..."] 也支持 {"primary": "..."}）
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, (list, tuple)):
                return parsed[0] if len(parsed) > 0 else ""
            if isinstance(parsed, dict):
                # 兼容名称/地址类字段
                return parsed.get("primary") or parsed.get("freeform") or ""
        except Exception:
            # 不是 JSON，就当作普通字符串返回
            return x
    # 其他类型（数字等）
    return str(x)

def extract_primary(x):
    # 解析 {"primary": "..."} / 或字符串本身
    if x is None or (isinstance(x, float) and pd.isna(x)): 
        return ""
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            if isinstance(obj, dict):
                return obj.get("primary", "")
        except Exception:
            return x
    if isinstance(x, dict):
        return x.get("primary", "")
    return str(x)

def extract_freeform_address(x):
    # 解析 [{"freeform": "..."}] 或 {"freeform": "..."}
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, str):
        try:
            obj = json.loads(x)
        except Exception:
            return x
    else:
        obj = x
    if isinstance(obj, list) and len(obj) > 0:
        v = obj[0]
        if isinstance(v, dict):
            return v.get("freeform", "")
        return str(v)
    if isinstance(obj, dict):
        return obj.get("freeform", "")
    return str(obj)


df = pd.read_parquet("datasets/project_c_samples_3k.parquet")

new_df = pd.DataFrame({
    "name":         df["names"].apply(extract_primary),
    "category":     df["categories"].apply(extract_primary),
    "address":      df["addresses"].apply(extract_freeform_address),
    "phone":        df["phones"].apply(first_item),
    "website":      df["websites"].apply(first_item),

    "base_name":        df["base_names"].apply(extract_primary),
    "base_category":    df["base_categories"].apply(extract_primary),
    "base_address":     df["base_addresses"].apply(extract_freeform_address),
    "base_phone":       df["base_phones"].apply(first_item),
    "base_website":     df["base_websites"].apply(first_item),
    "label":            df["label"].astype("Int64")
})
new_df.to_csv("overture_cleaned_places.csv", index=False)
print("Saved:", new_df.shape)

for col in ["phones","websites","base_phones","base_websites", "label"]:
    cnt = df[col].apply(lambda v: first_item(v) != "").sum()
    print(col, "non-empty after parsing:", cnt)
