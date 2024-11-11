import pandas as pd
import os
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def organize_files_by_label(csv_path, source_dir='.', destination_dir='sorted_files'):
    """
    Распределяет файлы по каталогам в соответствии с их метками из CSV файла.
    
    Args:
        csv_path (str): Путь к CSV файлу с данными
        source_dir (str): Директория с исходными файлами
        destination_dir (str): Директория для сортированных файлов
    """
    try:
        # Чтение CSV файла
        df = pd.read_csv(csv_path)
        logger.info(f"Успешно прочитан CSV файл: {csv_path}")
        
        # Создание корневой директории для сортированных файлов
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            logger.info(f"Создана корневая директория: {destination_dir}")
        
        # Получение уникальных меток
        labels = df['label'].unique()
        
        # Создание подкаталогов для каждой метки
        for label in labels:
            label_dir = os.path.join(destination_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
                logger.info(f"Создан каталог для метки {label}: {label_dir}")
        
        # Копирование файлов
        success_count = 0
        error_count = 0
        
        for _, row in df.iterrows():
            try:
                file_name = row['file_name']
                label = row['label']
                
                # Формирование путей
                source_path = os.path.join(source_dir, file_name)
                dest_dir = os.path.join(destination_dir, label)
                dest_path = os.path.join(dest_dir, os.path.basename(file_name))
                
                # Копирование файла
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    success_count += 1
                    logger.debug(f"Скопирован файл: {file_name} -> {dest_path}")
                else:
                    logger.warning(f"Файл не найден: {source_path}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Ошибка при копировании файла {file_name}: {str(e)}")
                error_count += 1
                
        logger.info(f"Завершено. Успешно скопировано файлов: {success_count}")
        if error_count > 0:
            logger.warning(f"Количество ошибок: {error_count}")
            
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    # Пример использования
    csv_file = "data_train.csv"
    source_directory = "."  # Текущая директория
    destination_directory = "data"
    
    organize_files_by_label(csv_file, source_directory, destination_directory)