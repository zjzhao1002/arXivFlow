import httpx
import pandas as pd
import pymupdf
import os
import asyncio
import datetime
import sqlite3
import gspread
from typing import List, Optional
from .ollama_functions import OllamaFunctions
from .gemini_functions import GeminiFunctions
from .categories import all_categories
from .arxiv_functions import DEFAULT_TIMEOUT, search_paper, download_pdf

class arXivFlow:

    def __init__(self, 
                 categories: str | List[str], 
                 start_date: str | datetime.datetime | None = None,
                 end_date: str | datetime.datetime | None = None,
                 max_results: Optional[int] = 100, 
                 ollama_model: Optional[str] = None,
                 gemini_model: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 request_timeout: float = DEFAULT_TIMEOUT
                 ):
        """
        The constructor of the arXivFlow class. This class is the orchestrator of the entire workflow: 
        - Querying arXiv for specific categories and date ranges.
        - Downloading PDFs.
        - Processing results and extracting information.
        - Saving data to CSV, JSON, Excel, SQLite, or Google Sheets.
        Args:
            categories: The category IDs (e.g., cs.AI, cs.LG, hep-ph) to be retrieved, which can be a string or a list of strings.
            start_date: The starting date in format 'YYYYMMDD' or the built-in datetime object: datetime(year, month, day). Defaults to 7 days (1 week) ago.
            end_date: The final date in format 'YYYYMMDD' or the built-in datetime object: datetime(year, month, day). Default is today (0 AM).
            max_results: The maximum number of results to retrieve. It can be set to None to retrieve all available data in the given date range, but the arXiv API's limit is 300,000 per query. Default is 100.
            ollama_model: The model name that can be run locally by Ollama. If the model is not available, the program will try to pull the model. If you don't need the LLM feature, you can set it to None, which is the default. 
            gemini_model: The Gemini model name to use. If both Ollama and Gemini are provided, Ollama takes precedence.
            gemini_api_key: API Key for Gemini. If not provided, it will read from the 'GOOGLE_AI_API' environment variable.
            request_timeout: Timeout in seconds for arXiv metadata and PDF requests. Default is 60 seconds.
        """
        self.categories = categories if isinstance(categories, list) else [categories]
        if not self.categories:
            raise ValueError("At least one arXiv category must be provided.")

        now = datetime.datetime.now()
        self.start_date = start_date if start_date is not None else now - datetime.timedelta(days=7)
        self.end_date = end_date if end_date is not None else now
        self.max_results = max_results
        self.ollama_model = ollama_model
        self.gemini_model = gemini_model
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_AI_API")
        if self.gemini_model is not None and self.ollama_model is None and not self.gemini_api_key:
            raise ValueError("Gemini API key is required when gemini_model is set. Pass gemini_api_key or set GOOGLE_AI_API.")
        self.client = httpx.AsyncClient(timeout=request_timeout, follow_redirects=True)
        self.dfs = []

    async def close(self):
        """
        Closes the underlying httpx client.
        """
        await self.client.aclose()

    def _get_date_string(self, date: datetime.datetime | str) -> str:
        """
        This function converts the date to 'YYYYMMDD000000' format, which set the time to midnight (00:00:00).
        Args: 
            date: A string in 'YYYYMMDD' format or a python built-in datetime object.
        Returns:
            A string in 'YYYYMMDD000000' format.
        """
        if isinstance(date, str):
            return date+"000000"
        return date.strftime("%Y%m%d000000")

    def _category_checker(self, category: str) -> str:
        """
        This function checks if the given category is available on arXiv (2026-05-01). See [arXiv: Category Taxonomy](https://arxiv.org/category_taxonomy)
        Args: 
            category: A string of the category ID.
        Returns:
            A string of the full name of the category if available.
        """
        if category not in all_categories.keys():
            raise ValueError(f"Invalid category: {category}. Please choose from the following categories: {', '.join(all_categories.keys())}")
        return all_categories[category]
    
    def set_pdfs_path(self, path: str) -> None:
        """
        This function sets the path to store downloaded PDF files. If the path does not exist, make one.
        Args:
            path: A string of the desired path.
        Returns:
            None
        """
        self.pdfs_path = path
        if not os.path.exists(self.pdfs_path):
            os.makedirs(self.pdfs_path)
    
    def _get_pdfs_path(self) -> str:
        """
        This function get the path to store downloaded PDF files. If the path is not specified, the program will make a directory called 'pdf_STARTDATE_to_ENDDATE' in the current directory.
        Args:
            None.
        Returns:
            pdf_dir: A string of the path to store PDF files.
        """
        if hasattr(self, 'pdfs_path') and self.pdfs_path:
            return self.pdfs_path
        else:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            current_dir = os.getcwd()
            pdf_dir = os.path.join(current_dir, f"pdf_{start_date[:8]}_to_{end_date[:8]}")
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir)
            return pdf_dir
    
    async def get_arxiv_data(self, download_pdfs: bool = False) -> pd.DataFrame:
        """
        This function fetches data of all given categories. Data are output to a pandas dataframe.
        Args:
            download_pdfs: A boolean variable to decide downloading PDF files or not.
        Returns:
            merged_df: A pandas dataframe with all fetched data. 
        """
        start_date = self._get_date_string(self.start_date)
        end_date = self._get_date_string(self.end_date)
        llm_functions = self._create_llm_functions()
        dfs = []
        for category in self.categories:
            df = await self._get_category_data(
                category,
                start_date,
                end_date,
                llm_functions=llm_functions,
                download_pdfs=False,
            )
            dfs.append(df)
        if download_pdfs:
            for df in dfs:
                await self._download_pdfs_for_dataframe(df, llm_functions=llm_functions)
        self.dfs = dfs
        merged_df = self._merged_dataframe()
        return merged_df

    async def _get_category_data(
            self,
            category: str,
            start_date: str,
            end_date: str,
            download_pdfs: bool = False,
            llm_functions: OllamaFunctions | GeminiFunctions | None = None
        ) -> pd.DataFrame:
        """
        This function fetches data of a specific category. Data are output to a pandas dataframe.
        Args:
            category: A string of the category ID to be fetched. 
            start_date: A string of the starting date in 'YYYYMMDD000000' format. 
            end_date: A string of the final date in 'YYYYMMDD000000' format. 
            download_pdfs: A boolean variable to decide downloading PDF files or not.
        Returns:
            df: A pandas dataframe with data of the given category.
        """
        if self._category_checker(category):
            print(f"Retrieving data for category {category} ({self._category_checker(category)}) from {start_date[:8]} to {end_date[:8]}...")

        query = f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"

        results = await search_paper(query=query, max_results=self._max_results_per_category(), client=self.client)

        data = []

        use_ollama = self.ollama_model is not None
        use_gemini = not use_ollama and self.gemini_model is not None

        for result in results: 
            # Download PDF
            if download_pdfs:
                try:
                    pdf_dir = self._get_pdfs_path()
                    pdf_url = result['PDF URL']
                    await download_pdf(pdf_url=pdf_url, dirpath=pdf_dir, filename=f"{result["arXiv ID"]}.pdf", client=self.client)
                    print(f"Downloaded PDF for {result["arXiv ID"]}")
                    if use_ollama:
                        text = await asyncio.to_thread(
                            self._extract_first_page_text,
                            f"{pdf_dir}/{result["arXiv ID"]}.pdf",
                        )
                        contact_info = await asyncio.to_thread(llm_functions.extract_contact_ollama, text) # type: ignore
                        print(f"Extracted Contact Information for {result["arXiv ID"]}.")
                    elif use_gemini:
                        text = await asyncio.to_thread(
                            self._extract_first_page_text,
                            f"{pdf_dir}/{result["arXiv ID"]}.pdf",
                        )
                        contact_info = await llm_functions.extract_contact_gemini(text) # type: ignore
                        print(f"Extracted Contact Information for {result["arXiv ID"]}.")
                    else:
                        contact_info = {"emails": [], "affiliations": []}
                except Exception as e:
                    print(f"Failed to download PDF for {result["arXiv ID"]}: {e}")
                    contact_info = {"emails": [], "affiliations": []}
            else:
                contact_info = {"emails": [], "affiliations": []}

            data.append(result)

            if download_pdfs and (use_ollama or use_gemini):
                emails = contact_info.get("emails", [])
                affiliations = contact_info.get("affiliations", [])
                result["Emails"] = ", ".join(emails) if emails else ""
                result["Affiliations"] = "; ".join(affiliations) if affiliations else ""

        df = pd.DataFrame(data)
        if df.empty:
            print(f"No results found for category {category} between {start_date[:8]} and {end_date[:8]}.")
            return df
        
        if use_ollama:
            keywords = [
                await asyncio.to_thread(
                    llm_functions.extract_keywords_ollama, # type: ignore
                    row["Title"],
                    row["Abstract"],
                )
                for _, row in df.iterrows()
            ]
            df['Keywords'] = keywords
            df["Keywords"] = df["Keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        elif use_gemini:
            keywords = [
                await llm_functions.extract_keywords_gemini( # type: ignore
                    row["Title"],
                    row["Abstract"],
                )
                for _, row in df.iterrows()
            ]
            df['Keywords'] = keywords
            df["Keywords"] = df["Keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        return df

    async def _download_pdfs_for_dataframe(
            self,
            df: pd.DataFrame,
            llm_functions: OllamaFunctions | GeminiFunctions | None = None
        ) -> pd.DataFrame:
        if df.empty:
            return df

        use_ollama = self.ollama_model is not None
        use_gemini = not use_ollama and self.gemini_model is not None

        if use_ollama or use_gemini:
            df["Emails"] = ""
            df["Affiliations"] = ""

        for index, row in df.iterrows():
            arxiv_id = row["arXiv ID"]
            try:
                pdf_dir = self._get_pdfs_path()
                pdf_path = await download_pdf(
                    pdf_url=row["PDF URL"],
                    dirpath=pdf_dir,
                    filename=f"{arxiv_id}.pdf",
                    client=self.client,
                )
                print(f"Downloaded PDF for {arxiv_id}")

                if use_ollama:
                    text = await asyncio.to_thread(self._extract_first_page_text, pdf_path)
                    contact_info = await asyncio.to_thread(llm_functions.extract_contact_ollama, text) # type: ignore
                    print(f"Extracted Contact Information for {arxiv_id}.")
                    emails = contact_info.get("emails", [])
                    affiliations = contact_info.get("affiliations", [])
                    df.at[index, "Emails"] = ", ".join(emails) if emails else ""
                    df.at[index, "Affiliations"] = "; ".join(affiliations) if affiliations else ""
                elif use_gemini:
                    text = await asyncio.to_thread(self._extract_first_page_text, pdf_path)
                    contact_info = await llm_functions.extract_contact_gemini(text) # type: ignore
                    print(f"Extracted Contact Information for {arxiv_id}.")
                    emails = contact_info.get("emails", [])
                    affiliations = contact_info.get("affiliations", [])
                    df.at[index, "Emails"] = ", ".join(emails) if emails else ""
                    df.at[index, "Affiliations"] = "; ".join(affiliations) if affiliations else ""
            except Exception as e:
                print(f"Failed to download PDF for {arxiv_id}: {e}")

        return df

    def _create_llm_functions(self) -> OllamaFunctions | GeminiFunctions | None:
        if self.ollama_model is not None:
            return OllamaFunctions(model_name=self.ollama_model)
        if self.gemini_model is not None:
            return GeminiFunctions(model_name=self.gemini_model, gemini_api_key=self.gemini_api_key) # type: ignore[arg-type]
        return None

    def _max_results_per_category(self) -> Optional[int]:
        if self.max_results is None:
            return None
        return max(1, (self.max_results + len(self.categories) - 1) // len(self.categories))

    def _extract_first_page_text(self, pdf_path: str) -> str:
        with pymupdf.open(pdf_path) as doc:
            if len(doc) == 0:
                return ""
            return doc[0].get_text() # type: ignore

    def _merged_dataframe(self) -> pd.DataFrame:
        if not self.dfs:
            return pd.DataFrame()
        non_empty_dfs = [df for df in self.dfs if not df.empty]
        if not non_empty_dfs:
            return pd.DataFrame()
        merged_df = pd.concat(non_empty_dfs, ignore_index=True)
        if "arXiv ID" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["arXiv ID"], keep="first")
        return merged_df.reset_index(drop=True)
    
    def save_to_csv(self, filename: Optional[str] = None) -> None:
        """
        This function saves data to a CSV file.
        Args:
            filename: The file name of the CSV file. If it is None, the program will generate a file called 'arxiv_data_STARTDATE_to_ENDDATE.csv' in the current directory. 
        Returns:
            None
        """
        merged_df = self._merged_dataframe()
        if filename is None:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            filename = f"arxiv_data_{start_date[:8]}_to_{end_date[:8]}.csv"
        merged_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_to_json(self, filename: Optional[str] = None) -> None:
        """
        This function saves data to a JSON file.
        Args:
            filename: The file name of the JSON file. If it is None, the program will generate a file called 'arxiv_data_STARTDATE_to_ENDDATE.json' in the current directory. 
        Returns:
            None
        """
        merged_df = self._merged_dataframe()
        if filename is None:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            filename = f"arxiv_data_{start_date[:8]}_to_{end_date[:8]}.json"
        merged_df.to_json(filename, orient='records', lines=True)
        print(f"Data saved to {filename}")

    def save_to_excel(self, filename: Optional[str] = None) -> None:
        """
        This function saves data to a EXCEL file.
        Args:
            filename: The file name of the EXCEL file. If it is None, the program will generate a file called 'arxiv_data_STARTDATE_to_ENDDATE.xlsx' in the current directory. 
        Returns:
            None
        """
        merged_df = self._merged_dataframe()
        if filename is None:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            filename = f"arxiv_data_{start_date[:8]}_to_{end_date[:8]}.xlsx"
        merged_df.to_excel(filename, index=False)
        print(f"Data saved to {filename}")

    def save_to_sqlite(self, filename: Optional[str] = None, table_name: str = "arxiv_data") -> None:
        """
        This function saves data to a SQL file.
        Args:
            filename: The file name of the SQL file. If it is None, the program will generate a file called 'arxiv_data_STARTDATE_to_ENDDATE.db' in the current directory. 
        Returns:
            None
        """
        merged_df = self._merged_dataframe()
        if filename is None:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            filename = f"arxiv_data_{start_date[:8]}_to_{end_date[:8]}.db"
        conn = sqlite3.connect(filename)
        merged_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Data saved to {filename} in table {table_name}")

    def save_to_google_sheet(self, sheet_id: str, credentials_file: str, sheet_name: Optional[str] = None) -> None:
        """
        This function uploads data to a Google Sheet. To use this function, a sheet ID and a Service Account are necessary. The target Google Sheet must be shared with the Service Account email.
        Args:
            sheet_id: The ID of the Google Sheet for saving data.
            credentials_file: The file name of the JSON key associated with a Service Account. 
            sheet_name: The sheet name to write data. Be careful that this function will clear existing data first. If it is None, this function will create a sheet named 'Data_STARTDATE_to_ENDDATE'.
        """
        merged_df = self._merged_dataframe()
        merged_df = merged_df.fillna('')  # Replace NaN with empty string for better compatibility with Google Sheets
        data = [merged_df.columns.values.tolist()] + merged_df.values.tolist()  # Convert DataFrame to list of lists
        if sheet_name is None:
            start_date = self._get_date_string(self.start_date)
            end_date = self._get_date_string(self.end_date)
            sheet_name = f"Data_{start_date[:8]}_to_{end_date[:8]}"
        try:
            gc = gspread.service_account(filename=credentials_file)
            workbook = gc.open_by_key(sheet_id)
            worksheets = workbook.worksheets()
            worksheet_titles = [ws.title for ws in worksheets]
            if sheet_name in worksheet_titles:
                worksheet = workbook.worksheet(sheet_name)
            else:
                worksheet = workbook.add_worksheet(title=sheet_name, rows=len(data), cols=len(data[0]) if data else 0)
            worksheet.clear()  # Clear existing content
            worksheet.update('A1', data)  # type: ignore # Update with new data starting from cell A1
            print("Google Sheet updated successfully.")
        except Exception as e:
            print(f"Error loading Google Sheets credentials: {e}")
            return
