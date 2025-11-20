import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Underwriting Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Underwriting Dashboard v2.0"
    }
)

# ==================== PROFESSIONAL STYLING ====================
st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .big-font {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .medium-font {
        font-size: 20px;
        font-weight: 600;
        color: #334155;
        margin-top: 1rem;
    }

    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        color: white;
        text-align: center;
    }

    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }

    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }

    .muted {
        color: #64748b;
        font-size: 14px;
    }

    .section-divider {
        border-top: 2px solid #e2e8f0;
        margin: 2rem 0;
    }

    .duplicate-warning {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== UTILITY FUNCTIONS ====================

@st.cache_data
def load_data_from_path(path: str):
    """Load CSV data from file path"""
    df = pd.read_csv(path, dtype=str)
    return df


@st.cache_data
def preprocess(df: pd.DataFrame):
    """Preprocess and clean the dataset"""
    df = df.rename(columns=lambda c: c.strip())

    if 'DOB' in df.columns:
        df['DOB_parsed'] = pd.to_datetime(df['DOB'], errors='coerce', dayfirst=False)
    else:
        df['DOB_parsed'] = pd.NaT

    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    if 'PolicyNumber' in df.columns:
        df['PolicyRoot'] = df['PolicyNumber'].astype(str).str.split('.', n=1, regex=False).str[0]
    else:
        df['PolicyRoot'] = np.nan

    if 'MembershipType' in df.columns:
        df['MembershipType'] = df['MembershipType'].str.strip().str.title()
    else:
        df['MembershipType'] = np.where(df['PolicyNumber'].str.contains(r'\.'), 'Dependent', 'Principal')

    if 'Status' in df.columns:
        df['Status'] = df['Status'].astype(str).str.strip().str.title()
    else:
        df['Status'] = 'Unknown'

    if 'RelationshipType' in df.columns:
        df['RelationshipType_norm'] = df['RelationshipType'].astype(str).str.lower()
    else:
        df['RelationshipType_norm'] = ''

    if 'PackageType' in df.columns:
        df['PackageType'] = df['PackageType'].astype(str).str.title()
    else:
        df['PackageType'] = 'Unknown'

    return df


def identify_duplicates(df: pd.DataFrame):
    """Identify different types of duplicates"""
    duplicates = {}

    # 1. Duplicate Policy Numbers
    dup_policy = df[df.duplicated(subset=['PolicyNumber'], keep=False)].sort_values('PolicyNumber')
    duplicates['policy_numbers'] = dup_policy

    # 2. Duplicate Principals (same PolicyRoot marked as Principal)
    principals_only = df[df['MembershipType'].str.lower() == 'principal']
    dup_principals = principals_only[principals_only.duplicated(subset=['PolicyRoot'], keep=False)].sort_values(
        'PolicyRoot')
    duplicates['principals'] = dup_principals

    # 3. Identical full rows
    dup_full = df[df.duplicated(keep=False)]
    duplicates['identical_rows'] = dup_full

    # 4. Duplicate Names (if Name column exists)
    if 'Name' in df.columns or 'FullName' in df.columns:
        name_col = 'Name' if 'Name' in df.columns else 'FullName'
        dup_names = df[df.duplicated(subset=[name_col], keep=False)].sort_values(name_col)
        duplicates['names'] = dup_names
    else:
        duplicates['names'] = pd.DataFrame()

    # 5. Potential duplicates (same DOB and similar other fields)
    if 'DOB_parsed' in df.columns and df['DOB_parsed'].notna().any():
        dup_dob = df[df.duplicated(subset=['DOB_parsed', 'Company'], keep=False)].sort_values('DOB_parsed')
        duplicates['dob_company'] = dup_dob
    else:
        duplicates['dob_company'] = pd.DataFrame()

    return duplicates


def principals_with_dependent_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count dependents per principal"""
    dependents = df[df['MembershipType'].str.lower() == 'dependent']
    dep_counts = dependents.groupby('PolicyRoot').size().reset_index(name='NumDependents')

    principals = df[df['MembershipType'].str.lower() == 'principal'][
        ['PolicyRoot', 'PolicyNumber', 'Company', 'PackageType', 'Status']
    ].drop_duplicates(subset=['PolicyRoot'])

    merged = principals.merge(dep_counts, on='PolicyRoot', how='left').fillna({'NumDependents': 0})
    merged['NumDependents'] = merged['NumDependents'].astype(int)
    return merged


def detect_spouse(dependents_df: pd.DataFrame):
    """Detect spouse dependents"""
    spouse_keywords = ['spouse', 'wife', 'husband', 'partner']
    mask = dependents_df['RelationshipType_norm'].fillna('').apply(
        lambda s: any(k in s for k in spouse_keywords)
    )
    return dependents_df[mask]


def age_on_date(dob, as_of_date):
    """Calculate age on a specific date"""
    if pd.isna(dob):
        return np.nan
    years = as_of_date.year - dob.year - (
            (as_of_date.month, as_of_date.day) < (dob.month, dob.day)
    )
    return years


def to_excel_bytes(df):
    """Convert DataFrame to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')

        workbook = writer.book
        worksheet = writer.sheets['Data']

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#3b82f6',
            'font_color': 'white',
            'border': 1
        })

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)

    return output.getvalue()


def create_pdf_report(df, title="Underwriting Report"):
    """Generate PDF report from DataFrame"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=30,
        alignment=1
    )
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.3 * inch))

    summary_style = styles['Normal']
    elements.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", summary_style))
    elements.append(Paragraph(f"<b>Total Records:</b> {len(df)}", summary_style))
    elements.append(Spacer(1, 0.3 * inch))

    df_subset = df.head(50)
    data = [df_subset.columns.tolist()] + df_subset.values.tolist()

    if len(data[0]) > 8:
        data = [row[:8] for row in data]
        elements.append(Paragraph("<i>Note: Showing first 8 columns only</i>", styles['Italic']))
        elements.append(Spacer(1, 0.2 * inch))

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==================== SIDEBAR & DATA LOADING ====================

st.sidebar.markdown("## 📊 Underwriting Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Status Analysis",
        "🔍 Principal Lookup",
        "👨‍👩‍👧‍👦 Dependents Analysis",
        "💑 Spouse & Relationship",
        "🎂 Age Filters",
        "📦 Package & Company",
        "⚠️ Duplicate Detection",
        "💾 Data & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Upload")
uploaded = st.sidebar.file_uploader("Upload CSV File", type=['csv'], help="Upload your underwriting dataset")

use_example = False
if uploaded is not None:
    df_raw = pd.read_csv(uploaded, dtype=str)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    try:
        df_raw = load_data_from_path('/mnt/data/underwriting dataset.csv')
        use_example = True
        st.sidebar.info("📁 Using default dataset")
    except Exception:
        df_raw = pd.DataFrame()
        st.sidebar.warning("⚠️ No dataset found. Please upload a CSV file.")

if df_raw.empty:
    st.error("❌ No data loaded. Please upload a CSV file to continue.")
    st.stop()

df = preprocess(df_raw)

# ==================== GLOBAL FILTERS ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Global Filters")

company_list = ['All'] + sorted(df['Company'].dropna().unique().tolist())
company_sel = st.sidebar.multiselect(
    "Company",
    options=company_list,
    default=['All'] if 'All' in company_list else company_list
)

if 'All' in company_sel:
    company_sel = [c for c in company_list if c != 'All']

package_list = ['All'] + sorted(df['PackageType'].dropna().unique().tolist())
package_sel = st.sidebar.multiselect(
    "Package Type",
    options=package_list,
    default=['All'] if 'All' in package_list else package_list
)

if 'All' in package_sel:
    package_sel = [p for p in package_list if p != 'All']

status_list = ['All'] + sorted(df['Status'].dropna().unique().tolist())
status_sel = st.sidebar.multiselect(
    "Status",
    options=status_list,
    default=['All'] if 'All' in status_list else status_list
)

if 'All' in status_sel:
    status_sel = [s for s in status_list if s != 'All']

membership_types = df['MembershipType'].dropna().unique().tolist()
member_sel = st.sidebar.multiselect(
    "Membership Type",
    options=membership_types,
    default=membership_types
)

# Apply filters
filtered = df[
    df['Company'].isin(company_sel) &
    df['PackageType'].isin(package_sel) &
    df['Status'].isin(status_sel) &
    df['MembershipType'].isin(member_sel)
    ].copy()

st.sidebar.markdown(f"**Filtered Records:** {len(filtered):,}")

# Quick duplicate alert in sidebar
duplicates_detected = identify_duplicates(df)
total_duplicate_issues = sum(len(v) for v in duplicates_detected.values() if not v.empty)
if total_duplicate_issues > 0:
    st.sidebar.markdown("---")
    st.sidebar.warning(f"⚠️ {total_duplicate_issues} duplicate records detected!")
    st.sidebar.markdown("Visit **Duplicate Detection** page for details")

# ==================== PAGES ====================

if page == "🏠 Overview":
    st.markdown("<div class='big-font'>📊 Underwriting Dashboard Overview</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Comprehensive analytics for underwriting portfolio management</p>",
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    total_policies = filtered['PolicyRoot'].nunique()
    total_rows = len(filtered)
    total_principals = filtered[filtered['MembershipType'].str.lower() == 'principal']['PolicyRoot'].nunique()
    total_dependents = filtered[filtered['MembershipType'].str.lower() == 'dependent']['PolicyNumber'].nunique()
    total_companies = filtered['Company'].nunique()

    col1.metric("📋 Total Policies", f"{total_policies:,}")
    col2.metric("👥 Total Members", f"{total_rows:,}")
    col3.metric("👤 Principals", f"{total_principals:,}")
    col4.metric("👨‍👩‍👧 Dependents", f"{total_dependents:,}")
    col5.metric("🏢 Companies", f"{total_companies:,}")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Package Distribution")
        pkg_counts = (
            filtered.drop_duplicates(subset=['PolicyRoot'])
            .groupby('PackageType')
            .size()
            .reset_index(name='Count')
        )
        fig1 = px.pie(
            pkg_counts,
            names='PackageType',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 🏢 Company Distribution")
        comp_counts = (
            filtered.drop_duplicates(subset=['PolicyRoot'])
            .groupby('Company')
            .size()
            .reset_index(name='Count')
            .sort_values('Count', ascending=False)
            .head(10)
        )
        fig2 = px.bar(
            comp_counts,
            x='Count',
            y='Company',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("#### 🎂 Age Distribution Analysis")

    today = date.today()
    filtered['AGE_FROM_DOB_TODAY'] = filtered['DOB_parsed'].apply(lambda d: age_on_date(d, today))
    age_series = filtered['AGE_FROM_DOB_TODAY'].dropna()

    if not age_series.empty:
        col1, col2 = st.columns([1, 2])

        with col1:
            age_bins = list(range(0, 101, 10))
            age_labels = [f"{age_bins[i]}-{age_bins[i + 1]}" for i in range(len(age_bins) - 1)]

            selected_group = st.selectbox(
                "Select Age Range",
                options=age_labels,
                help="Choose a range to view detailed age distribution"
            )

            st.metric("Average Age", f"{age_series.mean():.1f}")
            st.metric("Median Age", f"{age_series.median():.1f}")
            st.metric("Members with Age Data", f"{len(age_series):,}")

        with col2:
            start_age, end_age = map(int, selected_group.split('-'))
            ages_in_range = age_series[(age_series >= start_age) & (age_series <= end_age)]

            if not ages_in_range.empty:
                age_df = ages_in_range.value_counts().reset_index()
                age_df.columns = ['Age', 'Count']
                age_df = age_df.sort_values(by='Age')

                fig3 = px.bar(
                    age_df,
                    x="Age",
                    y="Count",
                    title=f"Age Distribution: {selected_group} years",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                fig3.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info(f"No ages recorded in the range {selected_group}")
    else:
        st.info("No DOB information available for age analysis")

elif page == "📈 Status Analysis":
    st.markdown("<div class='big-font'>📈 Policy Status Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Analyze policy statuses across your portfolio</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    status_counts = filtered['Status'].value_counts()
    active_count = status_counts.get('Active', 0)
    inactive_count = status_counts.get('Inactive', 0) + status_counts.get('Suspended', 0)
    total_status = len(filtered)

    col1.metric("✅ Active Policies", f"{active_count:,}")
    col2.metric("❌ Inactive Policies", f"{inactive_count:,}")
    col3.metric("📊 Active Rate", f"{(active_count / total_status * 100):.1f}%" if total_status > 0 else "N/A")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Status Distribution")
        status_df = filtered['Status'].value_counts().reset_index()
        status_df.columns = ['Status', 'Count']

        fig1 = px.pie(
            status_df,
            names='Status',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Status by Package Type")
        status_pkg = filtered.groupby(['PackageType', 'Status']).size().reset_index(name='Count')

        fig2 = px.bar(
            status_pkg,
            x='PackageType',
            y='Count',
            color='Status',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("#### 🏢 Status by Company")
    status_company = filtered.groupby(['Company', 'Status']).size().reset_index(name='Count')
    status_company_pivot = status_company.pivot(index='Company', columns='Status', values='Count').fillna(0)

    st.dataframe(status_company_pivot.style.background_gradient(cmap='Blues'), height=400)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        excel_data = to_excel_bytes(status_company_pivot.reset_index())
        st.download_button(
            "📥 Download Status Report (Excel)",
            data=excel_data,
            file_name=f"status_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        pdf_data = create_pdf_report(status_company_pivot.reset_index(), "Status Analysis Report")
        st.download_button(
            "📄 Download Status Report (PDF)",
            data=pdf_data,
            file_name=f"status_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

elif page == "🔍 Principal Lookup":
    st.markdown("<div class='big-font'>🔍 Principal Policy Lookup</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Search and analyze individual principal policies</p>", unsafe_allow_html=True)

    search_type = st.radio("Search by:", ["Policy Number", "Policy Root"], horizontal=True)

    if search_type == "Policy Number":
        policy_numbers = sorted(
            filtered[filtered['MembershipType'].str.lower() == 'principal']['PolicyNumber'].dropna().unique())

        if len(policy_numbers) > 0:
            search_term = st.text_input("Search Policy Number", placeholder="Type to search...")

            if search_term:
                filtered_policies = [p for p in policy_numbers if search_term.upper() in str(p).upper()]
            else:
                filtered_policies = policy_numbers[:100]

            selected_policy = st.selectbox(
                "Select Principal Policy Number",
                options=filtered_policies,
                help="Choose a principal policy to view details"
            )

            if selected_policy:
                principal_data = filtered[filtered['PolicyNumber'] == selected_policy]

                if not principal_data.empty:
                    principal_row = principal_data.iloc[0]

                    st.markdown("### 👤 Principal Information")
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Policy Number", principal_row['PolicyNumber'])
                    col2.metric("Company", principal_row['Company'])
                    col3.metric("Package", principal_row['PackageType'])
                    col4.metric("Status", principal_row['Status'])

                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

                    policy_root = principal_row['PolicyRoot']
                    all_dependents = filtered[
                        (filtered['PolicyRoot'] == policy_root) &
                        (filtered['MembershipType'].str.lower() == 'dependent')
                        ]

                    st.markdown("### 👨‍👩‍👧‍👦 Dependents Summary")
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Dependents", len(all_dependents))

                    spouse_count = detect_spouse(all_dependents)
                    col2.metric("Spouses", len(spouse_count))

                    children = all_dependents[
                        all_dependents['RelationshipType_norm'].str.contains('child|son|daughter', na=False)]
                    col3.metric("Children", len(children))

                    if not all_dependents.empty:
                        st.markdown("#### Dependents Details")
                        display_cols = ['PolicyNumber', 'RelationshipType', 'DOB_parsed', 'Status', 'PackageType']
                        available_cols = [col for col in display_cols if col in all_dependents.columns]
                        st.dataframe(all_dependents[available_cols].reset_index(drop=True), height=300)

                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        export_data = pd.concat([principal_data, all_dependents])

                        with col1:
                            excel_data = to_excel_bytes(export_data)
                            st.download_button(
                                "📥 Download Policy Data (Excel)",
                                data=excel_data,
                                file_name=f"policy_{selected_policy}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        with col2:
                            pdf_data = create_pdf_report(export_data, f"Policy Report: {selected_policy}")
                            st.download_button(
                                "📄 Download Policy Data (PDF)",
                                data=pdf_data,
                                file_name=f"policy_{selected_policy}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.info("No dependents found for this principal")
        else:
            st.warning("No principal policies found with current filters")

    else:  # Policy Root
        policy_roots = sorted(filtered['PolicyRoot'].dropna().unique())

        search_term = st.text_input("Search Policy Root", placeholder="Type to search...")

        if search_term:
            filtered_roots = [p for p in policy_roots if search_term.upper() in str(p).upper()]
        else:
            filtered_roots = policy_roots[:100]

        selected_root = st.selectbox("Select Policy Root", options=filtered_roots)

        if selected_root:
            root_data = filtered[filtered['PolicyRoot'] == selected_root]

            st.markdown("### Policy Family")
            st.metric("Total Members", len(root_data))
            st.dataframe(root_data.reset_index(drop=True))

            st.dataframe(root_data.reset_index(drop=True), height=400)

            # Summary metrics for the policy root
            principals_in_root = root_data[root_data['MembershipType'].str.lower() == 'principal']
            dependents_in_root = root_data[root_data['MembershipType'].str.lower() == 'dependent']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Principals", len(principals_in_root))
            col2.metric("Dependents", len(dependents_in_root))
            col3.metric("Companies", root_data['Company'].nunique())
            col4.metric("Packages", root_data['PackageType'].nunique())

elif page == "👨‍👩‍👧‍👦 Dependents Analysis":
    st.markdown("<div class='big-font'>👨‍👩‍👧‍👦 Dependents Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Comprehensive analysis of dependent relationships</p>", unsafe_allow_html=True)

    # Overall metrics
    dependents_df = filtered[filtered['MembershipType'].str.lower() == 'dependent']
    principals_with_deps = principals_with_dependent_counts(filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dependents", len(dependents_df))
    col2.metric("Principals with Dependents", len(principals_with_deps[principals_with_deps['NumDependents'] > 0]))
    col3.metric("Average Dependents/Principal", f"{principals_with_deps['NumDependents'].mean():.1f}")
    col4.metric("Max Dependents", principals_with_deps['NumDependents'].max())

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Dependents Distribution")
        dep_dist = principals_with_deps['NumDependents'].value_counts().sort_index().reset_index()
        dep_dist.columns = ['NumDependents', 'Count']

        fig = px.bar(
            dep_dist,
            x='NumDependents',
            y='Count',
            title="Number of Principals by Dependent Count",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 👥 Relationship Types")
        if 'RelationshipType' in dependents_df.columns:
            rel_counts = dependents_df['RelationshipType'].value_counts().head(10).reset_index()
            rel_counts.columns = ['Relationship', 'Count']

            fig = px.pie(
                rel_counts,
                names='Relationship',
                values='Count',
                title="Top 10 Relationship Types",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RelationshipType column available")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Dependents by Company
    st.markdown("#### 🏢 Dependents by Company")
    company_deps = dependents_df.groupby('Company').size().reset_index(name='Dependents')
    company_principals = filtered[filtered['MembershipType'].str.lower() == 'principal'].groupby(
        'Company').size().reset_index(name='Principals')

    company_analysis = company_principals.merge(company_deps, on='Company', how='left').fillna(0)
    company_analysis['Dependents/Principal'] = company_analysis['Dependents'] / company_analysis['Principals']
    company_analysis = company_analysis.sort_values('Dependents', ascending=False)

    st.dataframe(
        company_analysis.style.background_gradient(subset=['Dependents', 'Principals'], cmap='Blues')
        .background_gradient(subset=['Dependents/Principal'], cmap='YlOrRd'),
        height=400
    )

elif page == "💑 Spouse & Relationship":
    st.markdown("<div class='big-font'>💑 Spouse & Relationship Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Analyze spouse relationships and family structures</p>", unsafe_allow_html=True)

    dependents_df = filtered[filtered['MembershipType'].str.lower() == 'dependent']
    spouse_df = detect_spouse(dependents_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spouses", len(spouse_df))
    col2.metric("Spouse Percentage",
                f"{(len(spouse_df) / len(dependents_df) * 100):.1f}%" if len(dependents_df) > 0 else "0%")

    # Age analysis for spouses
    if 'DOB_parsed' in spouse_df.columns and spouse_df['DOB_parsed'].notna().any():
        today = date.today()
        spouse_df['Age'] = spouse_df['DOB_parsed'].apply(lambda d: age_on_date(d, today))
        avg_spouse_age = spouse_df['Age'].mean()
        col3.metric("Average Spouse Age", f"{avg_spouse_age:.1f}")
    else:
        avg_spouse_age = None
        col3.metric("Average Spouse Age", "N/A")

    col4.metric("Unique Policy Roots", spouse_df['PolicyRoot'].nunique())

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    if not spouse_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏢 Spouses by Company")
            spouse_company = spouse_df.groupby('Company').size().reset_index(name='Spouses')
            fig = px.bar(
                spouse_company,
                x='Spouses',
                y='Company',
                orientation='h',
                title="Spouse Count by Company",
                color='Spouses',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📦 Spouses by Package")
            spouse_package = spouse_df.groupby('PackageType').size().reset_index(name='Spouses')
            fig = px.pie(
                spouse_package,
                names='PackageType',
                values='Spouses',
                title="Spouse Distribution by Package",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Detailed spouse data
        st.markdown("#### 📋 Spouse Details")
        display_cols = ['PolicyNumber', 'PolicyRoot', 'Company', 'PackageType', 'RelationshipType', 'DOB_parsed',
                        'Status']
        available_cols = [col for col in display_cols if col in spouse_df.columns]
        st.dataframe(spouse_df[available_cols].reset_index(drop=True), height=400)

        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            excel_data = to_excel_bytes(spouse_df)
            st.download_button(
                "📥 Download Spouse Data (Excel)",
                data=excel_data,
                file_name=f"spouse_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            pdf_data = create_pdf_report(spouse_df, "Spouse Relationship Analysis")
            st.download_button(
                "📄 Download Spouse Data (PDF)",
                data=pdf_data,
                file_name=f"spouse_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("No spouse relationships detected in the current dataset")

elif page == "🎂 Age Filters":
    st.markdown("<div class='big-font'>🎂 Age Analysis & Filters</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Filter and analyze data by age ranges</p>", unsafe_allow_html=True)

    # Calculate ages
    today = date.today()
    filtered_with_age = filtered.copy()
    filtered_with_age['Age'] = filtered_with_age['DOB_parsed'].apply(lambda d: age_on_date(d, today))

    # Age range selector
    col1, col2, col3 = st.columns(3)

    with col1:
        min_age = st.slider("Minimum Age", 0, 100, 0)
    with col2:
        max_age = st.slider("Maximum Age", 0, 100, 100)
    with col3:
        st.metric("Available Records with Age", f"{filtered_with_age['Age'].notna().sum():,}")

    # Apply age filter
    age_filtered = filtered_with_age[
        (filtered_with_age['Age'] >= min_age) &
        (filtered_with_age['Age'] <= max_age) &
        (filtered_with_age['Age'].notna())
        ]

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Metrics for age-filtered data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records in Age Range", len(age_filtered))
    col2.metric("Average Age", f"{age_filtered['Age'].mean():.1f}" if len(age_filtered) > 0 else "N/A")
    col3.metric("Principals", len(age_filtered[age_filtered['MembershipType'].str.lower() == 'principal']))
    col4.metric("Dependents", len(age_filtered[age_filtered['MembershipType'].str.lower() == 'dependent']))

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Age Distribution")
        if not age_filtered.empty:
            fig = px.histogram(
                age_filtered,
                x='Age',
                nbins=20,
                title=f"Age Distribution ({min_age}-{max_age} years)",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data in selected age range")

    with col2:
        st.markdown("#### 👥 Membership Type by Age")
        if not age_filtered.empty:
            membership_age = age_filtered.groupby(
                ['MembershipType', pd.cut(age_filtered['Age'], bins=10)]).size().reset_index(name='Count')
            membership_age['Age Range'] = membership_age['Age'].astype(str)

            fig = px.bar(
                membership_age,
                x='Age Range',
                y='Count',
                color='MembershipType',
                title="Membership Type Distribution by Age",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data in selected age range")

    # Display filtered data
    if not age_filtered.empty:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Filtered Records")
        display_cols = ['PolicyNumber', 'MembershipType', 'Age', 'Company', 'PackageType', 'Status', 'DOB_parsed']
        available_cols = [col for col in display_cols if col in age_filtered.columns]
        st.dataframe(age_filtered[available_cols].reset_index(drop=True), height=400)

elif page == "📦 Package & Company":
    st.markdown("<div class='big-font'>📦 Package & Company Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Analyze package distribution across companies</p>", unsafe_allow_html=True)

    # Package-Company matrix
    package_company = filtered.groupby(['Company', 'PackageType']).size().reset_index(name='Count')
    package_company_pivot = package_company.pivot(index='Company', columns='PackageType', values='Count').fillna(0)

    st.markdown("#### 📈 Package Distribution by Company")
    st.dataframe(
        package_company_pivot.style.background_gradient(cmap='Blues', axis=1),
        height=500
    )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏢 Top Companies by Member Count")
        company_members = filtered.groupby('Company').size().reset_index(name='Members').sort_values('Members',
                                                                                                     ascending=False).head(
            10)

        fig = px.bar(
            company_members,
            x='Members',
            y='Company',
            orientation='h',
            title="Top 10 Companies by Member Count",
            color='Members',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📦 Package Popularity")
        package_counts = filtered.groupby('PackageType').size().reset_index(name='Count').sort_values('Count',
                                                                                                      ascending=False).head(
            10)

        fig = px.pie(
            package_counts,
            names='PackageType',
            values='Count',
            title="Top 10 Package Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Detailed analysis
    st.markdown("#### 📊 Package-Company Detailed View")
    selected_company = st.selectbox("Select Company for Detailed View", options=sorted(filtered['Company'].unique()))

    if selected_company:
        company_data = filtered[filtered['Company'] == selected_company]
        package_dist = company_data['PackageType'].value_counts().reset_index()
        package_dist.columns = ['PackageType', 'Count']

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Total Members", len(company_data))
            st.metric("Unique Packages", package_dist['PackageType'].nunique())
            st.metric("Principals", len(company_data[company_data['MembershipType'].str.lower() == 'principal']))
            st.metric("Dependents", len(company_data[company_data['MembershipType'].str.lower() == 'dependent']))

        with col2:
            fig = px.bar(
                package_dist,
                x='PackageType',
                y='Count',
                title=f"Package Distribution for {selected_company}",
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Duplicate Detection":
    st.markdown("<div class='big-font'>⚠️ Duplicate Detection</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Identify and manage duplicate records in your dataset</p>", unsafe_allow_html=True)

    duplicates = identify_duplicates(filtered)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duplicate Policy Numbers", len(duplicates['policy_numbers']))
    col2.metric("Duplicate Principals", len(duplicates['principals']))
    col3.metric("Identical Rows", len(duplicates['identical_rows']))
    col4.metric("Potential DOB Duplicates", len(duplicates['dob_company']))

    total_duplicates = sum(len(dup_df) for dup_df in duplicates.values() if not dup_df.empty)

    if total_duplicates > 0:
        st.markdown("<div class='duplicate-warning'>", unsafe_allow_html=True)
        st.warning(f"🚨 {total_duplicates} potential duplicate records detected!")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Duplicate type selector
    duplicate_type = st.selectbox(
        "Select Duplicate Type to Review:",
        [
            "Duplicate Policy Numbers",
            "Duplicate Principals",
            "Identical Rows",
            "Duplicate Names",
            "Potential DOB Duplicates"
        ]
    )

    # Display selected duplicate type
    if duplicate_type == "Duplicate Policy Numbers" and not duplicates['policy_numbers'].empty:
        st.markdown("#### 🔢 Duplicate Policy Numbers")
        st.dataframe(duplicates['policy_numbers'].reset_index(drop=True), height=400)

    elif duplicate_type == "Duplicate Principals" and not duplicates['principals'].empty:
        st.markdown("#### 👤 Duplicate Principals (same Policy Root)")
        st.dataframe(duplicates['principals'].reset_index(drop=True), height=400)

    elif duplicate_type == "Identical Rows" and not duplicates['identical_rows'].empty:
        st.markdown("#### 📋 Identical Rows")
        st.dataframe(duplicates['identical_rows'].reset_index(drop=True), height=400)

    elif duplicate_type == "Duplicate Names" and not duplicates['names'].empty:
        st.markdown("#### 📛 Duplicate Names")
        st.dataframe(duplicates['names'].reset_index(drop=True), height=400)

    elif duplicate_type == "Potential DOB Duplicates" and not duplicates['dob_company'].empty:
        st.markdown("#### 🎂 Potential DOB & Company Duplicates")
        st.dataframe(duplicates['dob_company'].reset_index(drop=True), height=400)

    else:
        st.info(f"No {duplicate_type.lower()} found in the current dataset")

    # Export duplicates
    if total_duplicates > 0:
        st.markdown("---")
        st.markdown("#### 📤 Export Duplicate Records")

        all_duplicates = pd.concat([df for df in duplicates.values() if not df.empty]).drop_duplicates()

        col1, col2 = st.columns(2)
        with col1:
            excel_data = to_excel_bytes(all_duplicates)
            st.download_button(
                "📥 Download All Duplicates (Excel)",
                data=excel_data,
                file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            pdf_data = create_pdf_report(all_duplicates, "Duplicate Records Report")
            st.download_button(
                "📄 Download All Duplicates (PDF)",
                data=pdf_data,
                file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

elif page == "💾 Data & Export":
    st.markdown("<div class='big-font'>💾 Data & Export</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Export filtered data and generate comprehensive reports</p>", unsafe_allow_html=True)

    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(filtered))
    col2.metric("Total Columns", len(filtered.columns))
    col3.metric("Data Size", f"{(filtered.memory_usage(deep=True).sum() / 1024 / 1024):.2f} MB")
    col4.metric("Missing Values", filtered.isnull().sum().sum())

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Column information
    st.markdown("#### 📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Column Information:**")
        column_info = pd.DataFrame({
            'Column': filtered.columns,
            'Data Type': filtered.dtypes.astype(str),
            'Non-Null Count': filtered.notna().sum(),
            'Null Count': filtered.isna().sum()
        })
        st.dataframe(column_info, height=300)

    with col2:
        st.markdown("**Quick Statistics:**")

        # Numeric columns summary
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("Numeric Columns:")
            st.dataframe(filtered[numeric_cols].describe(), height=200)
        else:
            st.info("No numeric columns in the dataset")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Export options
    st.markdown("#### 📤 Export Options")

    export_format = st.radio(
        "Select Export Format:",
        ["Excel", "CSV", "PDF Report"],
        horizontal=True
    )

    if export_format == "Excel":
        excel_data = to_excel_bytes(filtered)
        st.download_button(
            "📥 Download Excel File",
            data=excel_data,
            file_name=f"underwriting_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif export_format == "CSV":
        csv_data = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV File",
            data=csv_data,
            file_name=f"underwriting_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:  # PDF
        pdf_data = create_pdf_report(filtered, "Underwriting Data Export")
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_data,
            file_name=f"underwriting_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    # Advanced export options
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 🛠️ Advanced Export Options")

    col1, col2 = st.columns(2)

    with col1:
        # Export specific columns
        st.markdown("**Select Columns to Export:**")
        selected_columns = st.multiselect(
            "Choose columns:",
            options=filtered.columns.tolist(),
            default=filtered.columns.tolist()[:8]  # Default to first 8 columns
        )

    with col2:
        # Export filtered views
        st.markdown("**Export Specific Views:**")
        export_view = st.selectbox(
            "Select data view:",
            [
                "All Filtered Data",
                "Principals Only",
                "Dependents Only",
                "Active Policies",
                "Inactive Policies"
            ]
        )

    # Generate custom export
    if st.button("🔄 Generate Custom Export"):
        # Apply column selection
        custom_data = filtered[selected_columns] if selected_columns else filtered

        # Apply view filter
        if export_view == "Principals Only":
            custom_data = custom_data[custom_data['MembershipType'].str.lower() == 'principal']
        elif export_view == "Dependents Only":
            custom_data = custom_data[custom_data['MembershipType'].str.lower() == 'dependent']
        elif export_view == "Active Policies":
            custom_data = custom_data[custom_data['Status'].str.lower() == 'active']
        elif export_view == "Inactive Policies":
            custom_data = custom_data[custom_data['Status'].str.lower().isin(['inactive', 'suspended'])]

        st.success(f"Custom export ready: {len(custom_data)} records")

        # Download custom export
        custom_excel = to_excel_bytes(custom_data)
        st.download_button(
            "📥 Download Custom Export",
            data=custom_excel,
            file_name=f"custom_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 14px;'>"
    "Underwriting Analytics Platform v2.0 • Professional Dashboard for Portfolio Management"
    "</div>",
    unsafe_allow_html=True
)