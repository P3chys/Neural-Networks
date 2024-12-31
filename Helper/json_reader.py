
import json

class JsonReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def read(self):
        """Načte data z JSON souboru"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print(f"Soubor {self.file_path} nebyl nalezen.")
        except json.JSONDecodeError:
            print(f"Chyba při dekódování JSON souboru {self.file_path}.")
        except Exception as e:
            print(f"Došlo k chybě: {e}")

    def get_data(self):
        """Vrátí načtená data, pokud existují"""
        if self.data is not None:
            return self.data
        else:
            print("Data nebyla načtena.")
            return None

    def display_data(self):
        """Vypíše data načtená z JSON souboru"""
        if self.data is not None:
            print(json.dumps(self.data, indent=4, ensure_ascii=False))
        else:
            print("Data nejsou dostupná.")
