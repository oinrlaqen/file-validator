import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from rapidfuzz import fuzz, process

# load headers and test files
BASE_DIR = os.path.dirname(__file__)
validation_headers_path = os.path.join(BASE_DIR, "headers.csv")
messy_file_path = os.path.join(BASE_DIR, "messed_up_file.csv")
clean_file_path = os.path.join(BASE_DIR, "corrected_file.csv")

# ========= Expander with Sample Files ========= #

with st.expander("🧪 Try Sample Files"):

    col1, col2 = st.columns(2)

    # Read the local CSVs
    messy_file = pd.read_csv(messy_file_path)
    clean_file = pd.read_csv(clean_file_path)

    # Convert to in-memory CSV buffers
    messy_buffer = io.StringIO()
    messy_file.to_csv(messy_buffer, index = False)

    clean_buffer = io.StringIO()
    clean_file.to_csv(clean_buffer, index = False)

    # Download buttons
    with col1:
        st.download_button(
            label = "📥 Messed-Up File",
            data = messy_buffer.getvalue(),
            file_name = "messed_up_file.csv",
            mime = "text/csv",
            use_container_width = True
        )

    with col2:
        st.download_button(
            label = "📥 Corrected File",
            data = clean_buffer.getvalue(),
            file_name = "corrected_file.csv",
            mime = "text/csv",
            use_container_width = True
        )

# ========= Initialize App ========= #

st.title("Validate Your File")

uploaded_file = st.file_uploader("Choose CSV file", type = "csv")

# save original file name
original_file_name = uploaded_file.name if uploaded_file else "modified_file.csv"

if original_file_name.lower().endswith(".csv"):
    download_file_name = original_file_name
else:
    download_file_name = original_file_name.rsplit(".", 1)[0] + ".csv"

# function to clean dataframes from None values
def clean_nan_values(df: pd.DataFrame, add_row_offset: int = 2) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace([None, np.nan, "None", "none"], "")
    cleaned = cleaned.astype(str).applymap(
        lambda x: "" if x.strip().lower() in ["none", "nan"] else x.strip()
    )
    cleaned.index = cleaned.index + add_row_offset

    return cleaned

# clean visible None values from preview
if uploaded_file is not None:
    file = pd.read_csv(uploaded_file, index_col = False, dtype = str)

    st.subheader("Preview:")
    cleaned_review = clean_nan_values(file)

    st.dataframe(cleaned_review, use_container_width = True)

# store data for validation report
report_data = []

def add_to_report(message, df = None):
    report_data.append((message, df))

# ========= Functions for Headers Matching ========= #

def load_reference_map():
    reference_df = pd.read_csv(validation_headers_path)
    reference_map = {}

    for canonical in reference_df.columns:
        
        variants = reference_df[canonical].dropna().astype(str).str.strip().tolist()
        reference_map[canonical] = variants

    return reference_map

def match_headers(uploaded_headers, reference_map, threshold = 85):

    all_variants = {}
    for canonical, variants in reference_map.items():
        for var in variants:
            all_variants[var.lower()] = canonical

    best_matches = {canonical: ("", 0) for canonical in reference_map}

    for header in uploaded_headers:
        best_match, score, _ = process.extractOne(
            header.lower(),
            all_variants.keys(),
            scorer = fuzz.token_sort_ratio
        )

        if score >= threshold:
            canonical = all_variants[best_match]

            if score > best_matches[canonical][1]:
                best_matches[canonical] = (header, score)

    return {canonical: best_matches[canonical][0] for canonical in reference_map}

# =============== Check Header Duplicates =============== #

if uploaded_file is not None:
    
    uploaded_file.seek(0)
    headers = uploaded_file.readline().decode("utf-8").replace('\n', '').split(',')

    clean_headers = []
    duplicated_headers = []

    for header in headers:
        header_lower = header.strip()
        
        if header_lower == "":
            continue

        if header_lower in clean_headers:
            if header_lower not in duplicated_headers:
                duplicated_headers.append(header_lower)
        else:
            clean_headers.append(header_lower)

    if duplicated_headers:
        formatted = "\n".join(f"- {h}" for h in duplicated_headers)
        msg = f"⚠️ Duplicated column name(s) found: \n{formatted}"
        st.warning(msg)
        df_duplicated_headers = pd.DataFrame({"Duplicated Headers": duplicated_headers})
        add_to_report(msg)
    else:
        msg = "✅ No duplicated headers found"
        st.info(msg)
        add_to_report(msg)

    uploaded_file.seek(0)

# ================= Check Blank Headers ================= #

    blank_headers = []

    for i, header in enumerate(headers):
        if header.strip() == '':
            prev_header = headers[i - 1] if i > 0 else "<start>"
            blank_headers.append(prev_header)

    if blank_headers:
        formatted = "\n".join(f"- {h}" for h in blank_headers)
        msg = f"⚠️ Blank value(s) found after the following header(s): \n{formatted}"
        st.warning(msg)
        df_blank_headers = pd.DataFrame({"Blank Headers": blank_headers})
        add_to_report(msg)
    else:
        msg = "✅ No blank headers found"
        st.info(msg)
        add_to_report(msg)

# ========= Rename Columns if Necessary ========= #

    st.caption("""
    Check if all the mandatory fields are matched:

    - Employee ID  
    - First Name  
    - Last Name  
    - Relationship  
    - DOB  
    - SSN *(optional)*  
    - Medical / Dental / Vision Member ID *(optional)*  

    If you don't see these fields being matched in the table below, find them in the dropdown and rename them manually so they can be matched by the system.
    """)

    columns_to_rename = st.multiselect("Select columns to fix matchings:", file.columns)

    renamed_columns = {}
    for col in columns_to_rename:
        new_name = st.text_input(f"New name for '{col}'", value = col)
        renamed_columns[col] = new_name

    file = file.rename(columns = renamed_columns)
    updated_headers = file.columns.tolist()
    
# ============= Match Headers ============= #

    static_cols = [
        "Employee ID",
        "First Name",
        "Last Name",
        "Relationship",
        "Date of Birth",
        "SSN",
        "Medical Member ID",
        "Dental Member ID",
        "Vision Member ID"
    ]

    reference_map = load_reference_map()
    matched = match_headers(updated_headers, reference_map)

    st.subheader("Matched Headers:")
    
    rows = []
    for canonical in static_cols:
        matched_file_header = matched.get(canonical, "")
        rows.append((matched_file_header, canonical))

    matched_df = pd.DataFrame(rows, columns=["File", "Database"])
    
    st.dataframe(matched_df, use_container_width = True, hide_index = True)

# ============= Check Empty Values in Mandatory Fields ============= #

    st.divider()

    required_canonicals = {"Employee ID", "First Name", "Last Name", "Date of Birth", "Relationship"}
    selected_columns = [file_col for canon, file_col in matched.items() if canon in required_canonicals and file_col]

    with st.container(border = True):

        st.subheader("Empty Fields")

        if selected_columns:
            subset_df = file[selected_columns]
            nan_rows = subset_df[subset_df.isna().any(axis = 1)]

            if not nan_rows.empty:
                msg = f"⚠️ Found {len(nan_rows)} row(s) with missing values:"
                st.error(msg)

                nan_values_df = nan_rows.reset_index(drop = False)
                nan_values_df["Row Number"] = nan_values_df["index"] + 2
                nan_values_df = nan_values_df.drop(columns = ["index"])

                cols = ["Row Number"] + [col for col in nan_values_df.columns if col != "Row Number"]
                nan_values_df = nan_values_df[cols]

                clean_nan_values_df = nan_values_df.replace([None, np.nan, "None", "none"], "").astype(str)
                clean_nan_values_df = clean_nan_values_df.applymap(lambda x: "" if x.strip().lower() in ["none", "nan"] else x)

                st.dataframe(clean_nan_values_df.set_index("Row Number"), use_container_width = True)
                add_to_report(msg, clean_nan_values_df)
            else:
                msg = "✅ No missing values found in selected columns"
                st.info(msg)
                add_to_report(msg)
        else:
            msg = "⚠️ No matching columns found for required fields"
            st.warning(msg)
            add_to_report(msg)

# ============== Check if all the Deps have their Employee ============== #

    with st.container(border = True):
        st.subheader("All Dependents have their Employee")

        employee_col = next((file_col for canon, file_col in matched.items() if canon == "Employee ID" and file_col), None)
        relationship_col = next((file_col for canon, file_col in matched.items() if canon == "Relationship" and file_col), None)

        if not employee_col or not relationship_col:
            msg = "⚠️ Required columns 'Employee ID' and/or 'Relationship' were not found in the uploaded file"
            st.warning(msg)
            add_to_report(msg)
        else:
            if "employee_keywords" not in st.session_state:
                st.session_state["employee_keywords"] = {"employee", "self", "subscriber", "ee", "sub", "emp"}

            def match_relationship(value, threshold = 85):
                return any(fuzz.token_set_ratio(str(value).lower(), str(keyword).lower()) >= threshold 
                        for keyword in st.session_state["employee_keywords"])

            show_check = True
            show_warning = False  # Flag for external warning

            with st.expander("**Relationship Column Settings (click to expand)**"):
                unique_values = sorted(file[relationship_col].dropna().astype(str).str.strip().str.lower().unique())
                st.write("Relationship values in the file:", unique_values)

                new_keyword = st.text_input("**Add 'employee' value to the keywords if it's not there:**", placeholder="type here")
                if new_keyword:
                    st.session_state["employee_keywords"].add(new_keyword.strip().lower())

                st.markdown(f"Keywords:  `{', '.join(sorted(st.session_state['employee_keywords']))}`")

                matches_found = any(match_relationship(val) for val in unique_values)
                if not matches_found:
                    show_check = False
                    show_warning = True

            if show_warning:
                msg = "⚠️ Employee value not recognized. Add it to the keyword list"
                st.warning(msg)
                add_to_report(msg)

            if show_check:
                filtered_file = file[file[relationship_col].astype(str).str.strip().ne("")
                    & file[relationship_col].notna()]
                
                grouped = filtered_file.groupby(filtered_file[employee_col])[relationship_col].apply(list)

                invalid_ids = grouped[
                    grouped.apply(lambda x: any(not match_relationship(r) for r in x) and not any(match_relationship(r) for r in x))
                ]

                if not invalid_ids.empty:
                    msg = "⚠️ Some of the dependents do not have corresponding Employee record:"
                    st.error(msg)

                    bad_rows = file[file[employee_col].isin(invalid_ids.index)].copy()
                    bad_rows.insert(0, "Row Number", bad_rows.index + 2)

                    final_cols = ["Row Number"] + selected_columns

                    for col in selected_columns:
                        if col not in bad_rows.columns:
                            bad_rows[col] = ""
                    
                    bad_rows = bad_rows.reindex(columns=final_cols).reset_index(drop=True)

                    st.dataframe(clean_nan_values(bad_rows), use_container_width = True, hide_index = True)

                    add_to_report(msg, bad_rows)
                else:
                    msg = "✅ All of the dependents have corresponding Employee"
                    st.info(msg)
                    add_to_report(msg)

# ============= Check Invalid DOB ============= #

    with st.container(border = True):
        st.subheader("Invalid DOB")

        dob_col = next((file_col for canon, file_col in matched.items() if canon == "Date of Birth" and file_col), None)

        # -------- Check 1: Contains "9999" --------
        if not dob_col:
            msg = "⚠️ Date of Birth column not found in the uploaded file"
            st.warning(msg)
            add_to_report(msg)
        else:
            context_cols = [
                file_col for canon, file_col in matched.items()
                if canon in ["First Name", "Last Name", "Employee ID", "Relationship"] and file_col != dob_col
            ]

            dob_series_str = file[dob_col].fillna("").astype(str).str.strip()
            invalid_mask = dob_series_str.str.contains("9999")

            if invalid_mask.any():
                cols_to_show = [col for col in context_cols if col != dob_col] + [dob_col]
                invalid_rows_df = file.loc[invalid_mask, cols_to_show].copy()

                invalid_rows_df.insert(0, "Row Number", invalid_rows_df.index + 2)

                msg = f"⚠️ Found {len(invalid_rows_df)} row(s) where '{dob_col}' may contain '9999':"
                st.error(msg)
                st.dataframe(clean_nan_values(invalid_rows_df), use_container_width = True, hide_index = True)
                add_to_report(msg, invalid_rows_df)
            else:
                msg = "✅ No invalid DOB values containing '9999' found"
                st.info(msg)
                add_to_report(msg)
        
        # -------- Check 2: Future Dates --------
        if dob_col:
            context_cols = [
                file_col for canon, file_col in matched.items()
                if canon in ["First Name", "Last Name", "Employee ID", "Relationship"] and file_col != dob_col
            ]

            dob_series_str = file[dob_col].fillna("").astype(str).str.strip()

            dob_series_dt = pd.to_datetime(dob_series_str, format = 'mixed', errors = 'coerce')

            future_date_mask = dob_series_dt > datetime.now()

            if future_date_mask.any():
                cols_to_show = context_cols + [dob_col]
                future_rows_df = file.loc[future_date_mask, cols_to_show].copy()

                future_rows_df.insert(0, "Row Number", future_rows_df.index + 2)

                msg = f"⚠️ Found {len(future_rows_df)} row(s) with a future DOB in '{dob_col}':"
                st.error(msg)
                st.dataframe(clean_nan_values(future_rows_df), use_container_width=True, hide_index=True)
                add_to_report(msg, future_rows_df)
            else:
                msg = "✅ No future DOB values found"
                st.info(msg)
                add_to_report(msg)

# ============= Check SSN = Member ID ============= #

    with st.container(border=True):
        st.subheader("SSN = Member ID")

        ssn_column = next((file_col for canon, file_col in matched.items() if canon == "SSN" and file_col), None)
        member_id_columns = [file_col for canon, file_col in matched.items() if canon in {"Medical Member ID", "Dental Member ID", "Vision Member ID"} and file_col]

        if not ssn_column or not member_id_columns:
            msg = "⚠️ Could not automatically detect SSN and/or any Member ID columns"
            st.warning(msg)
            add_to_report(msg)
        else:
            def clean_column(s):
                s = s.fillna("").astype(str).str.replace(r"[^0-9]+", "", regex=True).str.strip()
                return s

            ssn_clean = clean_column(file[ssn_column]).reset_index(drop=True)
            found_any = False

            for member_col in member_id_columns:
                member_clean = clean_column(file[member_col]).reset_index(drop=True)

                match_mask = (
                    ssn_clean.str.fullmatch(r"\d{9}")
                    & member_clean.str.fullmatch(r"\d{9}")
                    & ~(ssn_clean.eq("") & member_clean.eq(""))
                    & ~(ssn_clean.isna() & member_clean.isna())
                    & (ssn_clean == member_clean))

                if match_mask.any():
                    found_any = True

                    final_cols = (["Row Number"]
                        + selected_columns
                        + [ssn_column, member_col])
                    
                    matched_rows = (
                        file.loc[match_mask, selected_columns + [ssn_column, member_col]]
                            .assign(**{"Row Number": lambda df: df.index + 2})
                            .pipe(clean_nan_values, add_row_offset=0)
                            .reindex(columns = final_cols))

                    msg = f"⚠️ Found {len(matched_rows)} row(s) where SSN matches '{member_col}':"
                    st.error(msg)
                    st.dataframe(clean_nan_values(matched_rows), use_container_width=True, hide_index=True)
                    add_to_report(msg, matched_rows)

            if not found_any:
                msg = "✅ No rows found where SSN matches any Member ID column"
                st.info(msg)
                add_to_report(msg)

# ============= Check Termination Date before Effective Date ============= #

    with st.container(border = True):

        st.subheader("Effective Date after Termination")

        for col in file.columns:
            if pd.api.types.is_numeric_dtype(file[col]):
                continue
            
            try:
                file[col] = pd.to_datetime(file[col], errors = 'raise')
            except (ValueError, TypeError):
                pass

        date_columns = file.select_dtypes(include = ['datetime64[ns]'])
        filtered_date_columns = [col for col in date_columns.columns if date_columns[col].notna().any()]

        with st.expander("**Date Columns in the File (click to expand)**"):
            st.write('''Columns with at least one date value listed below. If some columns need to be checked, 
                choose them side by side in the dropdowns under this list''', filtered_date_columns)

        if "dynamic_pairs" not in st.session_state:
            st.session_state.dynamic_pairs = []

        current_pairs = []

        all_options = [col for col in filtered_date_columns if col and col.strip()]

        used_columns = set()
        current_pairs = []

        max_pairs = len(all_options) // 2
        show_next = True

        for i in range(max_pairs):
            if not show_next:
                break

            available_options = [col for col in all_options if col not in used_columns]

            col1, col2 = st.columns(2)

            with col1:
                eff = st.selectbox(
                    f"Effective Column #{i+1}",
                    [""] + available_options,
                    key = f"eff_{i}"
                )

            with col2:
                remaining = [col for col in available_options if col != eff]
                term = st.selectbox(
                    f"Termination Column #{i+1}",
                    [""] + remaining,
                    key = f"term_{i}"
                )

            if eff and term and eff != term:
                pair = (eff, term)
                if pair not in current_pairs and (term, eff) not in current_pairs:
                    current_pairs.append(pair)
                    used_columns.update({eff, term})
                show_next = True
            else:
                show_next = False

        st.session_state.dynamic_pairs = current_pairs

        if st.session_state.dynamic_pairs:
            invalid_rows = []

            for eff_col, term_col in st.session_state.dynamic_pairs:
                eff_dates = pd.to_datetime(file[eff_col], errors = 'coerce')
                term_dates = pd.to_datetime(file[term_col], errors = 'coerce')

                valid_mask = eff_dates.notna()
                comparison_mask = eff_dates >= term_dates
                mask_invalid = comparison_mask & valid_mask

                invalid_count = mask_invalid.sum()
                valid_count = eff_dates.notna().sum()

                if invalid_count:
                    percent = (invalid_count / valid_count) * 100 if valid_count else 0
                    msg = (
                        f"⚠️ {invalid_count} row(s) affected "
                        f"({percent:.1f}% of {valid_count} '{eff_col}' values)"
                    )
                    st.error(msg)

                    final_cols = (["Row Number"]
                        + selected_columns
                        + [eff_col, term_col])

                    bad_rows = file.loc[mask_invalid, selected_columns + [eff_col, term_col]].copy()
                    bad_rows.insert(0, "Row Number", bad_rows.index + 2)
                    bad_rows = bad_rows.reindex(columns = final_cols).reset_index(drop = True) 

                    st.dataframe(clean_nan_values(bad_rows), use_container_width = True, hide_index = True)

                    add_to_report(msg, bad_rows)
                    invalid_rows.append((eff_col, term_col, invalid_count, percent))
                else:
                    msg = f"✅ Affected rows not found"
                    st.info(msg)
                    add_to_report(msg)

# ============= Download Validation Report & Updated CSV ============= #

    def create_validation_report(file_name, report_data):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine = "xlsxwriter") as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("Validation Report")
            writer.sheets["Validation Report"] = worksheet

            row_pos = 0

            for message, content in report_data:
                worksheet.write(row_pos, 0, f"{message}")
                row_pos += 2

                if content is not None:
                    if isinstance(content, pd.DataFrame):
                        content.to_excel(writer, sheet_name = "Validation Report", startrow = row_pos, index = False)
                        row_pos += len(content) + 2
                    elif isinstance(content, str):
                        for line in content.split("\n"):
                            worksheet.write(row_pos, 0, line)
                            row_pos += 1
                        row_pos += 1

        output.seek(0)
        return output

    report_file = create_validation_report(download_file_name, report_data)
    csv_buffer = file.to_csv(index = False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label = "📥 Download Validation Report",
            data = report_file,
            file_name = f"Validation Report - {download_file_name.rsplit('.', 1)[0]}.xlsx",
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label = "📥 Download Updated CSV",
            data = csv_buffer,
            file_name = download_file_name,
            mime = "text/csv",
            use_container_width = True
        )