import os
import openai
import copy
import glob
import shutil
openai.api_key = os.getenv("OPENAI_API_KEY")
from IPython.display import display, Code, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tiktoken

import numpy as np
import pandas as pd

import json
import io
import inspect
import requests
import re
import random
import string
import base64
import pymysql
import os.path
import matplotlib

from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseUpload
import base64
import email
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText
from openai.error import APIConnectionError

from bs4 import BeautifulSoup
import dateutil.parser as parser

import sys
from gptLearning import *
os.environ['SSL_VERSION'] = 'TLSv1_2'

import warnings
warnings.filterwarnings("ignore")

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from io import BytesIO


def create_or_get_folder(folder_name, upload_to_google_drive=False):
    """
    创建或获取文件夹ID，本地存储时获取文件夹路径
    """
    if upload_to_google_drive:
        # 若存储至谷歌云盘，则获取文件夹ID
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        # 查询是否已经存在该名称的文件夹
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        results = drive_service.files().list(q=query).execute()
        items = results.get('files', [])

        # 如果文件夹不存在，则创建它
        if not items:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = drive_service.files().create(body=folder_metadata).execute()
            folder_id = folder['id']
        else:
            folder_id = items[0]['id']
        
    else:
        # 若存储本地，则获取文件夹路径，且同时命名为folder_id
        folder_path = os.path.join('./', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_id = folder_path
        
    return folder_id

def create_or_get_doc(folder_id, doc_name, upload_to_google_drive=False):
    """
    创建或获取文件ID，本地存储时获取文件路径
    """    
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)
        docs_service = build('docs', 'v1', credentials=creds)

        # 查询文件夹中是否已经存在该名称的文档
        query = f"name='{doc_name}' and '{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        items = results.get('files', [])

        # 如果文档不存在，创建它
        if not items:
            doc_metadata = {
                'name': doc_name,
                'mimeType': 'application/vnd.google-apps.document',
                'parents': [folder_id]
            }
            doc = drive_service.files().create(body=doc_metadata).execute()
            document_id = doc['id']
        else:
            document_id = items[0]['id']
            
    # 若存储本地，则获取文件夹路径，且同时命名为document_id
    else: 
        file_path = os.path.join(folder_id, f'{doc_name}.md')
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('')  # 创建一个带有标题的空Markdown文件
        document_id = file_path
        
    return document_id

def get_file_content(file_id, upload_to_google_drive=False):
    """
    获取文档的具体内容，需要区分是读取谷歌云文档还是读取本地文档
    """
    # 读取谷歌云文档
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        service = build('drive', 'v3', credentials=creds)
        os.environ['SSL_VERSION'] = 'TLSv1_2'
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        content = request.execute()
        decoded_content = content.decode('utf-8')
        
    # 读取本地文档
    else:
        with open(file_id, 'r', encoding='utf-8') as file:
            decoded_content = file.read()
    return decoded_content

def append_content_in_doc(folder_id, doc_id, dict_list, upload_to_google_drive=False):
    """
    创建文档，或为指定的文档增加内容，需要区分是否是云文档
    """
    # 将字典列表转换为JSON字符串
    json_string = json.dumps(dict_list, indent=4, ensure_ascii=False)

    # 若是谷歌云文档
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)
        docs_service = build('docs', 'v1', credentials=creds)

        # 获取文档的当前长度
        document = docs_service.documents().get(documentId=doc_id).execute()
        end_of_doc = document['body']['content'][-1]['endIndex'] - 1  

        # 追加Q-A内容到文档
        requests = [{
            'insertText': {
                'location': {'index': end_of_doc},
                'text': json_string + '\n\n'   # 追加JSON字符串和两个换行，使格式整洁
            }
        }]
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
        
    # 若是本地文档
    else:
        with open(doc_id, 'a', encoding='utf-8') as file:
            file.write(json_string)  # 追加JSON字符串
            
def clear_content_in_doc(doc_id, upload_to_google_drive=False):
    """
    清空指定文档的全部内容，需要区分是否是云文档
    """
    # 如果是清除谷歌云文档内容
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        docs_service = build('docs', 'v1', credentials=creds)

        # 获取文档的当前长度
        document = docs_service.documents().get(documentId=doc_id).execute()
        end_of_doc = document['body']['content'][-1]['endIndex'] - 1

        # 创建删除内容的请求
        requests = [{
            'deleteContentRange': {
                'range': {
                    'startIndex': 1,  # 文档的开始位置
                    'endIndex': end_of_doc  # 文档的结束位置
                }
            }
        }]

        # 执行删除内容的请求
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
        
    # 如果是清除本地文档内容
    else:
        with open(doc_id, 'w') as file:
            pass  # 清空文件内容
        
def list_files_in_folder(folder_id, upload_to_google_drive=False):
    """
    列举当前文件夹的全部文件，需要区分是读取谷歌云盘文件夹还是本地文件夹
    """
    # 读取谷歌云盘文件夹内文件
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        # 列出文件夹中的所有文件
        query = f"'{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])

        # 获取并返回文件名称列表
        file_names = [file['name'] for file in files]
        
    # 读取本地文件夹内文件
    else:
        file_names = [f for f in os.listdir(folder_id) if os.path.isfile(os.path.join(folder_id, f))]
    return file_names

def rename_doc_in_drive(folder_id, doc_id, new_name, upload_to_google_drive=False):
    """
    修改指定的文档名称，需要区分是云文件还是本地文件
    """
    # 若修改云文档名称
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        # 创建更新请求以更改文档名称
        update_request_body = {
            'name': new_name
        }

        # 发送更新请求
        update_response = drive_service.files().update(
            fileId=doc_id,
            body=update_request_body,
            fields='id,name'
        ).execute()

        # 返回更新后的文档信息，包括ID和新名称
        update_name = update_response['name']
        
    # 若修改本地文档名称
    else:
        # 分解原始路径以获取目录和扩展名
        directory, old_file_name = os.path.split(doc_id)
        extension = os.path.splitext(old_file_name)[1]

        # 用新名称和原始扩展名组合新路径
        new_file_name = new_name + extension
        new_file_path = os.path.join(directory, new_file_name)

        # 重命名文件
        os.rename(doc_id, new_file_path)
        
        update_name=new_name
    
    return update_name

def delete_all_files_in_folder(folder_id, upload_to_google_drive=False):
    """
    删除某文件夹内全部文件，需要区分谷歌云文件夹还是本地文件夹
    """
    # 如果是谷歌云文件夹
    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        # 列出文件夹中的所有文件
        query = f"'{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])

        # 遍历并删除每个文件
        for file in files:
            file_id = file['id']
            drive_service.files().delete(fileId=file_id).execute()
            # print(f"Deleted file: {file['name']} (ID: {file_id})")
       
    # 如果是本地文件夹
    else:
        for filename in os.listdir(folder_id):
            file_path = os.path.join(folder_id, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                
class InterProject():
    """
    项目类：项目是每个分析任务的基础对象，换而言之，每个分析任务应该都是“挂靠”在某个项目中。\
    每个代码解释器必须说明所属项目，若无所属项目，则在代码解释器运行时会自动创建一个项目。\
    需要注意的是，项目不仅起到了说明和标注当前分析任务的作用，更关键的是，项目提供了每个分析任务的“长期记忆”，\
    即每个项目都有对应的谷歌云盘和谷歌云文档，用于保存在分析和建模工作过程中多轮对话内容，\
    此外，也可以选择借助本地文档进行存储。
    """

    def __init__(self, 
                 project_name, 
                 part_name, 
                 folder_id = None, 
                 doc_id = None, 
                 doc_content = None, 
                 upload_to_google_drive = False):
        
        # 项目名称，即项目文件夹名称
        self.project_name = project_name
        # 项目某部分名称，即项目文件名称
        self.part_name = part_name
        # 是否进行谷歌云文档存储
        self.upload_to_google_drive = upload_to_google_drive
        
        # 项目文件夹ID
        # 若项目文件夹ID为空，则获取项目文件夹ID
        if folder_id == None:
            folder_id = create_or_get_folder(folder_name=project_name,
                                             upload_to_google_drive = upload_to_google_drive)
        self.folder_id = folder_id
        
        # 创建时获取当前项目中其他文件名称列表
        self.doc_list = list_files_in_folder(folder_id, 
                                             upload_to_google_drive = upload_to_google_drive)
        
        # 项目文件ID
        # 若项目文件ID为空，则获取项目文件ID
        if doc_id == None:
            doc_id = create_or_get_doc(folder_id=folder_id, 
                                       doc_name=part_name, 
                                       upload_to_google_drive = upload_to_google_drive)
        self.doc_id = doc_id
        
        # 项目文件具体内容，相当于多轮对话内容
        self.doc_content = doc_content
        # 若初始content不为空，则将其追加入文档内
        if doc_content != None:
            append_content_in_doc(folder_id=folder_id, 
                                  doc_id=doc_id, 
                                  qa_string=doc_content, 
                                  upload_to_google_drive = upload_to_google_drive)
            

    def get_doc_content(self):
        """
        根据项目某文件的文件ID，获取对应的文件内容
        """     
        self.doc_content = get_file_content(file_id=self.doc_id, 
                                            upload_to_google_drive = self.upload_to_google_drive)

        return self.doc_content
    
    def append_doc_content(self, content):
        """
        根据项目某文件的文件ID，追加文件内容
        """  
        append_content_in_doc(folder_id=self.folder_id, 
                              doc_id=self.doc_id, 
                              dict_list=content, 
                              upload_to_google_drive = self.upload_to_google_drive)
    
    def clear_content(self):
        """
        清空某文件内的全部内容
        """  
        clear_content_in_doc(doc_id=self.doc_id, 
                             upload_to_google_drive = self.upload_to_google_drive)
        
    def delete_all_files(self):
        """
        删除当前项目文件夹内的全部文件
        """  
        delete_all_files_in_folder(folder_id=self.folder_id, 
                                   upload_to_google_drive = self.upload_to_google_drive)
        
    def update_doc_list(self):
        """
        更新当前项目文件夹内的全部文件名称
        """
        self.doc_list = list_files_in_folder(self.folder_id, 
                                             upload_to_google_drive = self.upload_to_google_drive)
    
    def rename_doc(self, new_name):
        """
        修改当前文件名称
        """
        self.part_name = rename_doc_in_drive(folder_id=self.folder_id, 
                                             doc_id=self.doc_id, 
                                             new_name=new_name, 
                                             upload_to_google_drive = self.upload_to_google_drive)
        
class ChatMessages():
    """
    ChatMessages类，用于创建Chat模型能够接收和解读的messages对象。该对象是原始Chat模型接收的\
    messages对象的更高级表现形式，ChatMessages类对象将字典类型的list作为其属性之一，同时还能\
    能区分系统消息和历史对话消息，并且能够自行计算当前对话的token量，并执能够在append的同时删\
    减最早对话消息，从而能够更加顺畅的输入大模型并完成多轮对话需求。
    """
    
    def __init__(self, 
                 system_content_list=[], 
                 question='你好。',
                 tokens_thr=None, 
                 project=None):

        self.system_content_list = system_content_list
        # 系统消息文档列表，相当于外部输入文档列表
        system_messages = []
        # 除系统消息外历史对话消息
        history_messages = []
        # 用于保存全部消息的list
        messages_all = []
        # 系统消息字符串
        system_content = ''
        # 历史消息字符串，此时为用户输入信息
        history_content = question
        # 系统消息+历史消息字符串
        content_all = ''
        # 输入到messages中系统消息个数，初始情况为0
        num_of_system_messages = 0
        # 全部信息的token数量
        all_tokens_count = 0
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 将外部输入文档列表依次保存为系统消息
        if system_content_list != []:      
            for content in system_content_list:
                system_messages.append({"role": "system", "content": content})
                # 同时进行全文档拼接
                system_content += content
                
            # 计算系统消息token
            system_tokens_count = len(encoding.encode(system_content))
            # 拼接系统消息
            messages_all += system_messages
            # 计算系统消息个数
            num_of_system_messages = len(system_content_list)
                
            # 若存在最大token数量限制
            if tokens_thr != None:
                # 若系统消息超出限制
                if system_tokens_count >= tokens_thr:
                    print("system_messages的tokens数量超出限制，当前系统消息将不会被输入模型，若有必要，请重新调整外部文档数量。")            
                    # 删除系统消息
                    system_messages = []
                    messages_all = []
                    # 系统消息个数清零
                    num_of_system_messages = 0
                    # 系统消息token数清零
                    system_tokens_count = 0
                    
            all_tokens_count += system_tokens_count
        
        # 创建首次对话消息
        history_messages = [{"role": "user", "content": question}]
        # 创建全部消息列表
        messages_all += history_messages
        
        # 计算用户问题token
        user_tokens_count = len(encoding.encode(question))
        
        # 计算总token数
        all_tokens_count += user_tokens_count
        
        # 若存在最大token限制
        if tokens_thr != None:
            # 若超出最大token限制
            if all_tokens_count >= tokens_thr:
                print("当前用户问题的tokens数量超出限制，该消息无法被输入到模型中，请重新输入用户问题或调整外部文档数量。")  
                # 同时清空系统消息和用户消息
                history_messages = []
                system_messages = []
                messages_all = []
                num_of_system_messages = 0
                all_tokens_count = 0
        
        # 全部messages信息
        self.messages = messages_all
        # system_messages信息
        self.system_messages = system_messages
        # user_messages信息
        self.history_messages = history_messages
        # messages信息中全部content的token数量
        self.tokens_count = all_tokens_count
        # 系统信息数量
        self.num_of_system_messages = num_of_system_messages
        # 最大token数量阈值
        self.tokens_thr = tokens_thr
        # token数计算编码方式
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # message挂靠的项目
        self.project = project
     
    # 删除部分对话信息
    def messages_pop(self, manual=False, index=None):
        def reduce_tokens(index):
            drop_message = self.history_messages.pop(index)
            self.tokens_count -= len(self.encoding.encode(str(drop_message)))

        if self.tokens_thr is not None:
            while self.tokens_count >= self.tokens_thr:
                reduce_tokens(-1)

        if manual:
            if index is None:
                reduce_tokens(-1)
            elif 0 <= index < len(self.history_messages) or index == -1:
                reduce_tokens(index)
            else:
                raise ValueError("Invalid index value: {}".format(index))

        # 更新messages
        self.messages = self.system_messages + self.history_messages
       
    # 增加部分对话信息
    def messages_append(self, new_messages):
        
        # 若是单独一个字典，或JSON格式字典
        if type(new_messages) is dict or type(new_messages) is openai.openai_object.OpenAIObject:
            self.messages.append(new_messages)
            self.tokens_count += len(self.encoding.encode(str(new_messages)))
            
        # 若新消息也是ChatMessages对象
        elif isinstance(new_messages, ChatMessages):
            self.messages += new_messages.messages
            self.tokens_count += new_messages.tokens_count

        # 重新更新history_messages
        self.history_messages = self.messages[self.num_of_system_messages: ]
        
        # 再执行pop，若有需要，则会删除部分历史消息
        self.messages_pop()
      
    # 复制信息
    def copy(self):
        # 创建一个新的 ChatMessages 对象，复制所有重要的属性
        system_content_str_list = [message['content'] for message in self.system_messages]
        new_obj = ChatMessages(
            system_content_list=copy.deepcopy(system_content_str_list),  # 使用深复制来复制系统消息
            question=self.history_messages[0]['content'] if self.history_messages else '',
            tokens_thr=self.tokens_thr
        )
        # 复制任何其他需要复制的属性
        new_obj.history_messages = copy.deepcopy(self.history_messages)  # 使用深复制来复制历史消息
        new_obj.messages = copy.deepcopy(self.messages)  # 使用深复制来复制所有消息
        new_obj.tokens_count = self.tokens_count
        new_obj.num_of_system_messages = self.num_of_system_messages
        
        return new_obj
    
    # 增加系统消息
    def add_system_messages(self, new_system_content):
        system_content_list = self.system_content_list
        system_messages = []
        # 若是字符串，则将其转化为list
        if type(new_system_content) == str:
            new_system_content = [new_system_content]
            
        system_content_list.extend(new_system_content)
        new_system_content_str = ''
        for content in new_system_content:
            new_system_content_str += content
        new_token_count = len(self.encoding.encode(str(new_system_content_str)))
        self.tokens_count += new_token_count
        self.system_content_list = system_content_list
        for message in system_content_list:
            system_messages.append({"role": "system", "content": message})
        self.system_messages = system_messages
        self.num_of_system_messages = len(system_content_list)
        self.messages = system_messages + self.history_messages
        
        # 再执行pop，若有需要，则会删除部分历史消息
        self.messages_pop()
        
        
    # 删除系统消息
    def delete_system_messages(self):
        system_content_list = self.system_content_list
        if system_content_list != []:
            system_content_str = ''
            for content in system_content_list:
                system_content_str += content
            delete_token_count = len(self.encoding.encode(str(system_content_str)))
            self.tokens_count -= delete_token_count
            self.num_of_system_messages = 0
            self.system_content_list = []
            self.system_messages = []
            self.messages = self.history_messages
     
    # 清除对话消息中的function消息
    def delete_function_messages(self):
        # 用于删除外部函数消息
        history_messages = self.history_messages
        # 从后向前迭代列表
        for index in range(len(history_messages) - 1, -1, -1):
            message = history_messages[index]
            if message.get("function_call") or message.get("role") == "function":
                self.messages_pop(manual=True, index=index)
                
def sql_inter(sql_query, g='globals()'):
    """
    用于执行一段SQL代码，并最终获取SQL代码执行结果，\
    核心功能是将输入的SQL代码传输至MySQL环境中进行运行，\
    并最终返回SQL代码运行结果。需要注意的是，本函数是借助pymysql来连接MySQL数据库。
    :param sql_query: 字符串形式的SQL查询语句，用于执行对MySQL中telco_db数据库中各张表进行查询，并获得各表中的各类相关信息
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：sql_query在MySQL中的运行结果。
    """
    
    mysql_pw = os.getenv('MYSQL_PW')
    
    connection = pymysql.connect(
            host='localhost',  # 数据库地址
            user='root',  # 数据库用户名
            passwd=mysql_pw,  # 数据库密码
            db='telco_db',  # 数据库名
            charset='utf8'  # 字符集选择utf8
        )
    
    try:
        with connection.cursor() as cursor:
            # SQL查询语句
            sql = sql_query
            cursor.execute(sql)

            # 获取查询结果
            results = cursor.fetchall()

    finally:
        connection.close()
    
    
    return json.dumps(results)

def extract_data(sql_query,df_name,g='globals()'):
    """
    借助pymysql将MySQL中的某张表读取并保存到本地Python环境中。
    :param sql_query: 字符串形式的SQL查询语句，用于提取MySQL中的某张表。
    :param df_name: 将MySQL数据库中提取的表格进行本地保存时的变量名，以字符串形式表示。
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：表格读取和保存结果
    """
    
    mysql_pw = os.getenv('MYSQL_PW')
    
    connection = pymysql.connect(
            host='localhost',  # 数据库地址
            user='root',  # 数据库用户名
            passwd=mysql_pw,  # 数据库密码
            db='telco_db',  # 数据库名
            charset='utf8'  # 字符集选择utf8
        )
    
    
    g[df_name] = pd.read_sql(sql_query, connection)
    
    return "已成功完成%s变量创建" % df_name

def python_inter(py_code, g='globals()'):
    """
    专门用于执行非绘图类python代码，并获取最终查询或处理结果。若是设计绘图操作的Python代码，则需要调用fig_inter函数来执行。
    :param py_code: 字符串形式的Python代码，用于执行对telco_db数据库中各张数据表进行操作
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果
    """    
    
    global_vars_before = set(g.keys())
    try:
        exec(py_code, g)            
    except Exception as e:
        return f"代码执行时报错{e}"
    global_vars_after = set(g.keys())
    new_vars = global_vars_after - global_vars_before
    # 若存在新变量
    if new_vars:
        result = {var: g[var] for var in new_vars}
        return str(result)
    # 若不存在新变量，即有可能是代码是表达式，也有可能代码对相同变量重复赋值
    else:
        try:
            # 尝试如果是表达式，则返回表达式运行结果
            return str(eval(py_code, g))
        # 若报错，则先测试是否是对相同变量重复赋值
        except Exception as e:
            try:
                exec(py_code, g)
                return "已经顺利执行代码"
            except Exception as e:
                pass
            # 若不是重复赋值，则报错
            return f"代码执行时报错{e}"
        
def upload_image_to_drive(figure, folder_id = '1YstWRU-78JwTEQQA3vJokK3OF_F0djRH'):
    """
    将指定的fig对象上传至谷歌云盘
    """
    folder_id = folder_id        # 此处需要改为自己的谷歌云盘文件夹ID
    creds = Credentials.from_authorized_user_file('token.json')
    drive_service = build('drive', 'v3', credentials=creds)
    
    # 1. Save image to Google Drive
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
    file_metadata = {
        'name': 'YourImageName.png',
        'parents': [folder_id],
        'mimeType': 'image/png'
    }
    image_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id,webContentLink'  # Specify the fields to be returned
    ).execute()
    
    return image_file["webContentLink"]

def fig_inter(py_code, fname, g='globals()'):
    """
    用于执行一段包含可视化绘图的Python代码，并最终获取一个图片类型对象
    :param py_code: 字符串形式的Python代码，用于根据需求进行绘图，代码中必须包含Figure对象创建过程
    :param fname: py_code代码中创建的Figure变量名，以字符串形式表示。
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果
    """    
    # 保存当前的后端
    current_backend = matplotlib.get_backend()
    
    # 设置为Agg后端
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # 创建一个字典，用于存储本地变量
    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    
    try:
        exec(py_code, g, local_vars)       
    except Exception as e:
        return f"代码执行时报错{e}"
    
    # 回复默认后端
    matplotlib.use(current_backend)
    
    # 根据图片名称，获取图片对象
    fig = local_vars[fname]
    
    # 上传图片
    try:
        fig_url = upload_image_to_drive(fig)
        res = f"已经成功运行代码，并已将代码创建的图片存储至：{fig_url}"
        
    except Exception as e:
        res = "无法上传图片至谷歌云盘，请检查谷歌云盘文件夹ID，并检查当前网络情况"
        
    print(res)
    return res

def auto_functions(functions_list):
    """
    Chat模型的functions参数编写函数
    :param functions_list: 包含一个或者多个函数对象的列表；
    :return：满足Chat模型functions参数要求的functions对象
    """
    def functions_generate(functions_list):
        # 创建空列表，用于保存每个函数的描述字典
        functions = []
        
        def chen_ming_algorithm(data):
            """
            陈明算法函数，该函数定义了一种特殊的数据集计算过程
            :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
            :return：陈明函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
            """
            df_new = pd.read_json(data)
            res = np.sum(df_new, axis=1) - 1
            return res.to_json(orient='records')
        
        chen_ming_function_description = inspect.getdoc(chen_ming_algorithm)
        
        chen_ming_function_name = chen_ming_algorithm.__name__
        
        chen_ming_function = {"name": "chen_ming_algorithm",
                              "description": "用于执行陈明算法的函数，定义了一种特殊的数据集计算过程",
                              "parameters": {"type": "object",
                                             "properties": {"data": {"type": "string",
                                                                     "description": "执行陈明算法的数据集"},
                                                           },
                                             "required": ["data"],
                                            },
                             }

        
        # 对每个外部函数进行循环
        for function in functions_list:
            # 读取函数对象的函数说明
            function_description = inspect.getdoc(function)
            # 读取函数的函数名字符串
            function_name = function.__name__

            user_message1 = '以下是某的函数说明：%s。' % chen_ming_function_description +\
                            '根据这个函数的函数说明，请帮我创建一个function对象，用于描述这个函数的基本情况。这个function对象是一个JSON格式的字典，\
                            这个字典有如下5点要求：\
                            1.字典总共有三个键值对；\
                            2.第一个键值对的Key是字符串name，value是该函数的名字：%s，也是字符串；\
                            3.第二个键值对的Key是字符串description，value是该函数的函数的功能说明，也是字符串；\
                            4.第三个键值对的Key是字符串parameters，value是一个JSON Schema对象，用于说明该函数的参数输入规范。\
                            5.输出结果必须是一个JSON格式的字典，只输出这个字典即可，前后不需要任何前后修饰或说明的语句' % chen_ming_function_name
            
            
            assistant_message1 = json.dumps(chen_ming_function)
            
            user_prompt = '现在有另一个函数，函数名为：%s；函数说明为：%s；\
                          请帮我仿造类似的格式为当前函数创建一个function对象。' % (function_name, function_description)

            response = openai.ChatCompletion.create(
                              model="gpt-4-0613",
                              messages=[
                                {"role": "user", "name":"example_user", "content": user_message1},
                                {"role": "assistant", "name":"example_assistant", "content": assistant_message1},
                                {"role": "user", "name":"example_user", "content": user_prompt}]
                            )
            functions.append(json.loads(response.choices[0].message['content']))
        return functions
    
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            functions = functions_generate(functions_list)
            break  # 如果代码成功执行，跳出循环
        except Exception as e:
            attempts += 1  # 增加尝试次数
            print("发生错误：", e)
            print("由于模型limit rate导致报错，即将暂停1分钟，1分钟后重新尝试调用模型")
            time.sleep(60)
            
            if attempts == max_attempts:
                print("已达到最大尝试次数，程序终止。")
                raise  # 重新引发最后一个异常
            else:
                print("正在重新运行...")
    return functions

class AvailableFunctions():
    """
    外部函数类，主要负责承接外部函数调用时相关功能支持。类属性包括外部函数列表、外部函数参数说明列表、以及调用方式说明三项。
    """
    def __init__(self, functions_list=[], functions=[], function_call="auto"):
        self.functions_list = functions_list
        self.functions = functions
        self.functions_dic = None
        self.function_call = None
        # 当外部函数列表不为空、且外部函数参数解释为空时，调用auto_functions创建外部函数解释列表
        if functions_list != []:
            self.functions_dic = {func.__name__: func for func in functions_list}
            self.function_call = function_call
            if functions == []:
                self.functions = auto_functions(functions_list)
       
    # 增加外部函数方法，并且同时可以更换外部函数调用规则
    def add_function(self, new_function, function_description=None, function_call_update=None):
        self.functions_list.append(new_function)
        self.functions_dic[new_function.__name__] = new_function
        if function_description == None:
            new_function_description = auto_functions([new_function])
            self.functions.append(new_function_description)
        else:
            self.functions.append(function_description)
        if function_call_update != None:
            self.function_call = function_call_update
            
def add_task_decomposition_prompt(messages):
    
    """
    当开启增强模式时，任何问题首次尝试作答时都会调用本函数，创建一个包含任务拆解Few-shot的新的message。
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认值为None，表示不存在外部函数。
    :return: task_decomp_few_shot，一个包含任务拆解Few-shot提示示例的message
    """
    
    # 任务拆解Few-shot
    # 第一个提示示例
    user_question1 = '请问谷歌云邮箱是什么？'
    user_message1_content = "现有用户问题如下：“%s”。为了回答这个问题，总共需要分几步来执行呢？\
    若无需拆分执行步骤，请直接回答原始问题。" % user_question1
    assistant_message1_content = '谷歌云邮箱是指Google Workspace（原G Suite）中的Gmail服务，\
    它是一个安全、智能、易用的电子邮箱，有15GB的免费存储空间，可以直接在电子邮件中接收和存储邮件。\
    Gmail 邮箱会自动过滤垃圾邮件和病毒邮件，并且可以通过电脑或手机等移动设备在任何地方查阅邮件。\
    您可以使用搜索和标签功能来组织邮件，使邮件处理更为高效。'

    # 第二个提示示例
    user_question2 = '请帮我介绍下OpenAI。'
    user_message2_content = "现有用户问题如下：“%s”。为了回答这个问题，总共需要分几步来执行呢？\
    若无需拆分执行步骤，请直接回答原始问题。" % user_question2
    assistant_message2_content = 'OpenAI是一家开发和应用友好人工智能的公司，\
    它的目标是确保人工通用智能（AGI）对所有人都有益，以及随着AGI部署，尽可能多的人都能受益。\
    OpenAI致力在商业利益和人类福祉之间做出正确的平衡，本质上是一家人道主义公司。\
    OpenAI开发了诸如GPT-3这样的先进模型，在自然语言处理等诸多领域表现出色。'

    # 第三个提示示例
    user_question3 = '围绕数据库中的user_payments表，我想要检查该表是否存在缺失值'
    user_message3_content = "现有用户问题如下：“%s”。为了回答这个问题，总共需要分几步来执行呢？\
    若无需拆分执行步骤，请直接回答原始问题。" % user_question3
    assistant_message3_content = '为了检查user_payments数据集是否存在缺失值，我们将执行如下步骤：\
    \n\n步骤1：使用`extract_data`函数将user_payments数据表读取到当前的Python环境中。\
    \n\n步骤2：使用`python_inter`函数执行Python代码检查数据集的缺失值。'

    # 第四个提示示例
    user_question4 =  '我想寻找合适的缺失值填补方法，来填补user_payments数据集中的缺失值。'
    user_message4_content = "现有用户问题如下：“%s”。为了回答这个问题，总共需要分几步来执行呢？\
    若无需拆分执行步骤，请直接回答原始问题。" % user_question4
    assistant_message4_content = '为了找到合适的缺失值填充方法，我们需要执行以下三步：\
    \n\n步骤1：分析user_payments数据集中的缺失值情况。通过查看各字段的缺失率和观察缺失值分布，了解其缺失幅度和模式。\
    \n\n步骤2：确定值填补策略。基于观察结果和特定字段的性质确定恰当的填补策略，例如使用众数、中位数、均值或建立模型进行填补等。\
    \n\n步骤3：进行缺失值填补。根据确定的填补策略，执行填补操作，然后验证填补效果。'
    
    # 在保留原始问题的情况下加入Few-shot
    task_decomp_few_shot = messages.copy()
    task_decomp_few_shot.messages_pop(manual=True, index=-1)
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message1_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message1_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message2_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message2_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message3_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message3_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message4_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message4_content})
    
    user_question = messages.history_messages[-1]["content"]

    new_question = "现有用户问题如下：“%s”。为了回答这个问题，总共需要分几步来执行呢？\
    若无需拆分执行步骤，请直接回答原始问题。" % user_question
    question_message = messages.history_messages[-1].copy()
    question_message["content"] = new_question
    task_decomp_few_shot.messages_append(question_message)
    
    return task_decomp_few_shot

def modify_prompt(messages, action='add', enable_md_output=True, enable_COT=True):
    """
    当开启开发者模式时，会让用户选择是否添加COT提示模板或其他提示模板，并创建一个经过修改的新的message。
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param action: 'add' 或 'remove'，决定是添加还是移除提示
    :param enable_md_output: 是否启用 markdown 格式输出
    :param enable_COT: 是否启用 COT 提示
    :return: messages，一个经过提示词修改的message
    """
    
    # 思考链提示词模板
    cot_prompt = "请一步步思考并得出结论。"
    
    # 输出markdown提示词模板
    md_prompt = "任何回答都请以markdown格式进行输出。"

    # 如果是添加提示词
    if action == 'add':
        if enable_COT:
            messages.messages[-1]["content"] += cot_prompt
            messages.history_messages[-1]["content"] += cot_prompt

        if enable_md_output:
            messages.messages[-1]["content"] += md_prompt
            messages.history_messages[-1]["content"] += md_prompt
       
    # 如果是将指定提示词删除
    elif action == 'remove':
        if enable_md_output:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(md_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(md_prompt, "")
        
        if enable_COT:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(cot_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(cot_prompt, "")

    return messages

def get_gpt_response(model, 
                     messages, 
                     available_functions=None,
                     is_developer_mode=False,
                     is_enhanced_mode=False):
    
    """
    负责调用Chat模型并获得模型回答函数，并且当在调用GPT模型时遇到Rate limit时可以选择暂时休眠1分钟后再运行。\
    同时对于意图不清的问题，会提示用户修改输入的prompt，以获得更好的模型运行结果。
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认为None，表示没有外部函数
    :param is_developer_mode: 表示是否开启开发者模式，默认为False。\
    开启开发者模式时，会自动添加提示词模板，并且会在每次执行代码前、以及返回结果之后询问用户意见，并会根据用户意见进行修改。
    :param is_enhanced_mode: 可选参数，表示是否开启增强模式，默认为False。\
    开启增强模式时，会自动启动复杂任务拆解流程，并且在进行代码debug时会自动执行deep debug。
    :return: 返回模型返回的response message
    """
    
    # 如果开启开发者模式，则进行提示词修改，首次运行是增加提示词
    if is_developer_mode:
        messages = modify_prompt(messages, action='add')
        
    # 如果是增强模式，则增加复杂任务拆解流程
    # if is_enhanced_mode:
        # messages = add_task_decomposition_prompt(messages)

    # 考虑到可能存在通信报错问题，因此循环调用Chat模型进行执行
    while True:
        try:
            # 若不存在外部函数
            if available_functions == None:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages)   
                
            # 若存在外部函数，此时functions和function_call参数信息都从AvailableFunctions对象中获取
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages, 
                    functions=available_functions.functions, 
                    function_call=available_functions.function_call
                    )   
            break  # 如果成功获取响应，退出循环
            
        except APIConnectionError as e:
            # APIConnectionError默认是用户需求不清导致无法返回结果
            # 若开启增强模式，此时提示用户重新输入需求
            if is_enhanced_mode:
                # 创建临时消息列表
                msg_temp = messages.copy()
                # 获取用户问题
                question = msg_temp.messages[-1]["content"]
                # 提醒用户修改提问的提示模板
                new_prompt = "以下是用户提问：%s。该问题有些复杂，且用户意图并不清晰。\
                请编写一段话，来引导用户重新提问。" % question
                # 修改msg_temp并重新提问
                try:
                    msg_temp.messages[-1]["content"] = new_prompt
                    # 修改用户问题并直接提问
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=msg_temp.messages)
                    
                    # 打印gpt返回的提示修改原问题的描述语句
                    display(Markdown(response["choices"][0]["message"]["content"]))
                    # 引导用户重新输入问题或者退出
                    user_input = input("请重新输入问题，输入“退出”可以退出当前对话")
                    if user_input == "退出":
                        print("当前模型无法返回结果，已经退出")
                        return None
                    else:
                        # 修改原始问题
                        messages.history_messages[-1]["content"] = user_input
                        
                        # 再次进行提问
                        response_message = get_gpt_response(model=model, 
                                                            messages=messages, 
                                                            available_functions=available_functions,
                                                            is_developer_mode=is_developer_mode,
                                                            is_enhanced_mode=is_enhanced_mode)
                        
                        return response_message
                # 若在提示用户修改原问题时遇到链接错误，则直接暂停1分钟后继续执行While循环
                except APIConnectionError as e:
                    print(f"当前遇到了一个链接问题: {str(e)}")
                    print("由于Limit Rate限制，即将等待1分钟后继续运行...")
                    time.sleep(60)  # 等待1分钟
                    print("已等待60秒，即将开始重新调用模型并进行回答...")
            
            # 若未开启增强模式       
            else:        
                # 打印错误的核心信息
                print(f"当前遇到了一个链接问题: {str(e)}")
                # 如果是开发者模式
                if is_developer_mode:
                    # 选择等待、更改模型或者直接报错退出
                    user_input = input("请选择等待1分钟（1），或者更换模型（2），或者报错退出（3）")
                    if user_input == '1':
                        print("好的，将等待1分钟后继续运行...")
                        time.sleep(60)  # 等待1分钟
                        print("已等待60秒，即将开始新的一轮问答...")
                    elif user_input == '2':
                        model = input("好的，请输出新模型名称")
                    else:
                        if modify:
                            messages = modify_prompt(messages, action='remove', enable_md_output=md_output, enable_COT=COT)
                        raise e  # 如果用户选择退出，恢复提示并抛出异常
                # 如果不是开发者模式
                else:
                    print("由于Limit Rate限制，即将等待1分钟后继续运行...")
                    time.sleep(60)  # 等待1分钟
                    print("已等待60秒，即将开始重新调用模型并进行回答...")

    # 还原原始的msge对象
    if is_developer_mode:
        messages = modify_prompt(messages, action='remove')
        
    return response["choices"][0]["message"]

def get_chat_response(model, 
                      messages, 
                      available_functions=None,
                      is_developer_mode=False,
                      is_enhanced_mode=False, 
                      delete_some_messages=False, 
                      is_task_decomposition=False):
    
    """
    负责完整执行一次对话的最高层函数，需要注意的是，一次对话中可能会多次调用大模型，而本函数则是完成一次对话的主函数。\
    要求输入的messages中最后一条消息必须是能正常发起对话的消息。\
    该函数通过调用get_gpt_response来获取模型输出结果，并且会根据返回结果的不同，例如是文本结果还是代码结果，\
    灵活调用不同函数对模型输出结果进行后处理。\
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认为None，表示没有外部函数
    :param is_developer_mode: 表示是否开启开发者模式，默认为False。\
    开启开发者模式时，会自动添加提示词模板，并且会在每次执行代码前、以及返回结果之后询问用户意见，并会根据用户意见进行修改。
    :param is_enhanced_mode: 可选参数，表示是否开启增强模式，默认为False。\
    开启增强模式时，会自动启动复杂任务拆解流程，并且在进行代码debug时会自动执行deep debug。
    :param delete_some_messages: 可选参数，表示在拼接messages时是否删除中间若干条消息，默认为Fasle。
    :param is_task_decomposition: 可选参数，是否是当前执行任务是否是审查任务拆解结果，默认为False。
    :return: 拼接本次问答最终结果的messages
    """
    
    # 当且仅当围绕复杂任务拆解结果进行修改时，才会出现is_task_decomposition=True的情况
    # 当is_task_decomposition=True时，不再重新创建response_message
    if not is_task_decomposition:
        # 先获取单次大模型调用结果
        # 此时response_message是大模型调用返回的message
        response_message = get_gpt_response(model=model, 
                                            messages=messages, 
                                            available_functions=available_functions,
                                            is_developer_mode=is_developer_mode,
                                            is_enhanced_mode=is_enhanced_mode)
    
    # 复杂条件判断，若is_task_decomposition = True，
    # 或者是增强模式且是执行function response任务时
    # （需要注意的是，当is_task_decomposition = True时，并不存在response_message对象）
    if is_task_decomposition or (is_enhanced_mode and response_message.get("function_call")):
        # 将is_task_decomposition修改为True，表示当前执行任务为复杂任务拆解
        is_task_decomposition = True
        # 在拆解任务时，将增加了任务拆解的few-shot-message命名为text_response_messages
        task_decomp_few_shot = add_task_decomposition_prompt(messages)
        # print("正在进行任务分解，请稍后...")
        # 同时更新response_message，此时response_message就是任务拆解之后的response
        response_message = get_gpt_response(model=model, 
                                            messages=task_decomp_few_shot, 
                                            available_functions=available_functions,
                                            is_developer_mode=is_developer_mode,
                                            is_enhanced_mode=is_enhanced_mode)
        # 若拆分任务的提示无效，此时response_message有可能会再次创建一个function call message
        if response_message.get("function_call"):
            print("当前任务无需拆解，可以直接运行。")

    # 若本次调用是由修改对话需求产生，则按照参数设置删除原始message中的若干条消息
    # 需要注意的是，删除中间若干条消息，必须在创建完新的response_message之后再执行
    if delete_some_messages:
        for i in range(delete_some_messages):
            messages.messages_pop(manual=True, index=-1)
    
    # 注意，执行到此处时，一定会有一个response_message
    # 接下来分response_message不同类型，执行不同流程
    # 若是文本响应类任务（包括普通文本响应和和复杂任务拆解审查两种情况，都可以使用相同代码）
    if not response_message.get("function_call"):
        # 将message保存为text_answer_message
        text_answer_message = response_message 
        # 并带入is_text_response_valid对文本内容进行审查
        messages = is_text_response_valid(model=model, 
                                          messages=messages, 
                                          text_answer_message=text_answer_message,
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=is_enhanced_mode, 
                                          delete_some_messages=delete_some_messages,
                                          is_task_decomposition=is_task_decomposition)
    
    
    
    # 若是function response任务
    elif response_message.get("function_call"):
        # 创建调用外部函数的function_call_message
        # 在当前Agent中，function_call_message是一个包含SQL代码或者Python代码的JSON对象
        function_call_message = response_message 
        # 将function_call_message带入代码审查和运行函数is_code_response_valid
        # 并最终获得外部函数运行之后的问答结果
        messages = is_code_response_valid(model=model, 
                                          messages=messages, 
                                          function_call_message=function_call_message,
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=is_enhanced_mode, 
                                          delete_some_messages=delete_some_messages)
    
    return messages    

# 判断代码输出结果是否符合要求，输入function call message，输出function response message
def is_code_response_valid(model, 
                           messages, 
                           function_call_message,
                           available_functions=None,
                           is_developer_mode=False,
                           is_enhanced_mode=False, 
                           delete_some_messages=False):
    
    
    """
    负责完整执行一次外部函数调用的最高层函数，要求输入的msg最后一条消息必须是包含function call的消息。\
    函数的最终任务是将function call的消息中的代码带入外部函数并完成代码运行，并且支持交互式代码编写或自动代码编写运行不同模式。\
    当函数运行得到一条包含外部函数运行结果的function message之后，会继续将其带入check_get_final_function_response函数，\
    用于最终将function message转化为assistant message，并完成本次对话。
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param function_call_message: 必要参数，用于表示上层函数创建的一条包含function call消息的message
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认为None，表示没有外部函数
    :param is_developer_mode: 表示是否开启开发者模式，默认为False。\
    开启开发者模式时，会自动添加提示词模板，并且会在每次执行代码前、以及返回结果之后询问用户意见，并会根据用户意见进行修改。
    :param is_enhanced_mode: 可选参数，表示是否开启增强模式，默认为False。\
    开启增强模式时，会自动启动复杂任务拆解流程，并且在进行代码debug时会自动执行deep debug。
    :param delete_some_messages: 可选参数，表示在拼接messages时是否删除中间若干条消息，默认为Fasle。
    :return: message，拼接了最新大模型回答结果的message
    """
    
    # 为打印代码和修改代码（增加创建图像对家部分代码）做准备
    # 创建字符串类型json格式的message对象
    code_json_str = function_call_message["function_call"]["arguments"]
    # 将json转化为字典
    try:
        code_dict = json.loads(code_json_str)
    except Exception as e:
        print("json字符解析错误，正在重新创建代码...")
        # 递归调用上层函数get_chat_response，并返回最终message结果
        # 需要注意的是，如果上层函数再次创建了function_call_message
        # 则会再次调用is_code_response_valid，而无需在当前函数中再次执行
        messages = get_chat_response(model=model, 
                                     messages=messages, 
                                     available_functions=available_functions,
                                     is_developer_mode=is_developer_mode,
                                     is_enhanced_mode=is_enhanced_mode, 
                                     delete_some_messages=delete_some_messages)
        
        return messages
        
    # 若顺利将json转化为字典，则继续执行以下代码
    # 创建convert_to_markdown内部函数，用于辅助打印代码结果
    def convert_to_markdown(code, language):
        return f"```{language}\n{code}\n```"

    # 提取代码部分参数
    # 如果是SQL，则按照Markdown中SQL格式打印代码
    if code_dict.get('sql_query'):
        code = code_dict['sql_query'] 
        markdown_code = convert_to_markdown(code, 'sql')
        print("即将执行以下代码：")
        
    # 如果是Python，则按照Markdown中Python格式打印代码
    elif code_dict.get('py_code'):
        code = code_dict['py_code']
        markdown_code = convert_to_markdown(code, 'python')
        print("即将执行以下代码：")
        
    else:
        markdown_code = code_dict
        
    display(Markdown(markdown_code))
        
      
    # 若是开发者模式，则提示用户先对代码进行审查然后再运行
    if is_developer_mode:         
        user_input = input("是直接运行代码（1），还是反馈修改意见，并让模型对代码进行修改后再运行（2）")
        if user_input == '1':
            print("好的，正在运行代码，请稍后...")
                
        else:
            modify_input = input("好的，请输入修改意见：")
            # 记录模型当前创建的代码
            messages.messages_append(function_call_message)
            # 记录修改意见
            messages.messages_append({"role": "user", "content": modify_input})
            
            # 调用get_chat_response函数并重新获取回答结果
            # 需要注意，此时需要设置delete_some_messages=2，删除中间对话结果以节省token
            messages = get_chat_response(model=model, 
                                         messages=messages, 
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode, 
                                         delete_some_messages=2)
            
            return messages
                
    # 若不是开发者模式，或者开发者模式下user_input == '1'
    # 则调用function_to_call函数，并获取最终外部函数运行结果
    # 在当前Agent中，外部函数运行结果就是SQL或者Python运行结果，或代码运行报错结果
    function_response_message = function_to_call(available_functions=available_functions, 
                                                 function_call_message=function_call_message)  
    
    # 将function_response_message带入check_get_final_function_response进行审查
    messages = check_get_final_function_response(model=model, 
                                                 messages=messages, 
                                                 function_call_message=function_call_message,
                                                 function_response_message=function_response_message,
                                                 available_functions=available_functions,
                                                 is_developer_mode=is_developer_mode,
                                                 is_enhanced_mode=is_enhanced_mode, 
                                                 delete_some_messages=delete_some_messages)
    
    return messages

# 判断代码输出结果是否符合要求，输入function response message，输出基于外部函数运行结果的message
def check_get_final_function_response(model, 
                                      messages, 
                                      function_call_message,
                                      function_response_message,
                                      available_functions=None,
                                      is_developer_mode=False,
                                      is_enhanced_mode=False, 
                                      delete_some_messages=False):
    
    """
    负责执行外部函数运行结果审查工作。若外部函数运行结果消息function_response_message并不存在报错信息，\
    则将其拼接入message中，并将其带入get_chat_response函数并获取下一轮对话结果。而如果function_response_message中存在报错信息，\
    则开启自动debug模式。本函数将借助类似Autogen的模式，复制多个Agent，并通过彼此对话的方式来完成debug。
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param function_call_message: 必要参数，用于表示上层函数创建的一条包含function call消息的message
    :param function_response_message: 必要参数，用于表示上层函数创建的一条包含外部函数运行结果的message
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认为None，表示没有外部函数
    :param is_developer_mode: 表示是否开启开发者模式，默认为False。\
    开启开发者模式时，会自动添加提示词模板，并且会在每次执行代码前、以及返回结果之后询问用户意见，并会根据用户意见进行修改。
    :param is_enhanced_mode: 可选参数，表示是否开启增强模式，默认为False。\
    开启增强模式时，会自动启动复杂任务拆解流程，并且在进行代码debug时会自动执行deep debug。
    :param delete_some_messages: 可选参数，表示在拼接messages时是否删除中间若干条消息，默认为Fasle。
    :return: message，拼接了最新大模型回答结果的message
    """    
    
    # 获取外部函数运行结果内容
    fun_res_content = function_response_message["content"]
    
    # 若function_response中包含错误
    if "报错" in fun_res_content:
        # 打印报错信息
        print(fun_res_content)
        
        # 根据是否是增强模式，选择执行高效debug或深度debug
        # 高效debug和深度debug区别只在于提示内容和提示流程的不同
        # 高效debug只包含一条提示，只调用一次大模型即可完成自动debug工作
        # 而深度debug则包含三次提示，需要调用三次大模型进行深度总结并完成debug工作
        # 先创建不同模式bubug的不同提示词
        if not is_enhanced_mode:
            # 执行高效debug
            display(Markdown("**即将执行高效debug，正在实例化Efficient Debug Agent...**"))
            debug_prompt_list = ['你编写的代码报错了，请根据报错信息修改代码并重新执行。']
            
        else:
            # 执行深度debug
            display(Markdown("**即将执行深度debug，该debug过程将自动执行多轮对话，请耐心等待。正在实例化Deep Debug Agent...**"))
            display(Markdown("**正在实例化deep debug Agent...**"))
            debug_prompt_list = ["之前执行的代码报错了，你觉得代码哪里编写错了？", 
                                 "好的。那么根据你的分析，为了解决这个错误，从理论上来说，应该如何操作呢？", 
                                 "非常好，接下来请按照你的逻辑编写相应代码并运行。"]
        
        # 复制msg，相当于创建一个新的Agent进行debug
        # 需要注意的是，此时msg最后一条消息是user message，而不是任何函数调用相关message
        msg_debug = messages.copy()        
        # 追加function_call_message
        # 当前function_call_message中包含编错的代码
        msg_debug.messages_append(function_call_message)
        # 追加function_response_message
        # 当前function_response_message包含错误代码的运行报错信息
        msg_debug.messages_append(function_response_message)        
        
        # 依次输入debug的prompt，来引导大模型完成debug
        for debug_prompt in debug_prompt_list:
            msg_debug.messages_append({"role": "user", "content": debug_prompt})
            display(Markdown("**From Debug Agent:**"))
            display(Markdown(debug_prompt))
            
            # 再次调用get_chat_response，在当前debug的prompt下，get_chat_response会返回修改意见或修改之后的代码
            # 打印提示信息
            display(Markdown("**From MateGen:**"))
            msg_debug = get_chat_response(model=model, 
                                          messages=msg_debug, 
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=False, 
                                          delete_some_messages=delete_some_messages)
        
        messages = msg_debug.copy()     
                 
    # 若function message不包含报错信息    
    # 需要将function message传递给模型
    else:
        print("外部函数已执行完毕，正在解析运行结果...")
        messages.messages_append(function_call_message)
        messages.messages_append(function_response_message)
        messages = get_chat_response(model=model, 
                                     messages=messages, 
                                     available_functions=available_functions,
                                     is_developer_mode=is_developer_mode,
                                     is_enhanced_mode=is_enhanced_mode, 
                                     delete_some_messages=delete_some_messages)
        
    return messages

def is_text_response_valid(model, 
                           messages, 
                           text_answer_message,
                           available_functions=None,
                           is_developer_mode=False,
                           is_enhanced_mode=False, 
                           delete_some_messages=False,
                           is_task_decomposition=False):
    
    """
    负责执行文本内容创建审查工作。运行模式可分为快速模式和人工审查模式。在快速模式下，模型将迅速创建文本并保存至msg对象中，\
    而如果是人工审查模式，则需要先经过人工确认，函数才会保存大模型创建的文本内容，并且在这个过程中，\
    也可以选择让模型根据用户输入的修改意见重新修改文本。
    :param model: 必要参数，表示调用的大模型名称
    :param messages: 必要参数，ChatMessages类型对象，用于存储对话消息
    :param text_answer_message: 必要参数，用于表示上层函数创建的一条包含文本内容的message
    :param available_functions: 可选参数，AvailableFunctions类型对象，用于表示开启对话时外部函数基本情况。\
    默认为None，表示没有外部函数
    :param is_developer_mode: 表示是否开启开发者模式，默认为False。\
    开启开发者模式时，会自动添加提示词模板，并且会在每次执行代码前、以及返回结果之后询问用户意见，并会根据用户意见进行修改。
    :param is_enhanced_mode: 可选参数，表示是否开启增强模式，默认为False。\
    开启增强模式时，会自动启动复杂任务拆解流程，并且在进行代码debug时会自动执行deep debug。
    :param delete_some_messages: 可选参数，表示在拼接messages时是否删除中间若干条消息，默认为Fasle。
    :param is_task_decomposition: 可选参数，是否是当前执行任务是否是审查任务拆解结果，默认为False。
    :return: message，拼接了最新大模型回答结果的message
    """    
    
    # 从text_answer_message中获取模型回答结果并打印
    answer_content = text_answer_message["content"]
    
    print("模型回答：\n")
    display(Markdown(answer_content))
    
    # 创建指示变量user_input，用于记录用户修改意见，默认为None
    user_input = None
    
    # 若是开发者模式，或者是增强模式下任务拆解结果，则引导用户对其进行审查
    # 若是开发者模式而非任务拆解
    if not is_task_decomposition and is_developer_mode:
        user_input = input("请问是否记录回答结果（1），\
        或者对当前结果提出修改意见（2），\
        或者重新进行提问（3），\
        或者直接退出对话（4）")
        if user_input == '1':
            # 若记录回答结果，则将其添加入msg对象中
            messages.messages_append(text_answer_message)
            print("本次对话结果已保存")
        
    # 若是任务拆解
    elif is_task_decomposition:
        user_input = input("请问是否按照该流程执行任务（1），\
        或者对当前执行流程提出修改意见（2），\
        或者重新进行提问（3），\
        或者直接退出对话（4）")
        if user_input == '1':
            # 任务拆解中，如果选择执行该流程
            messages.messages_append(text_answer_message)
            print("好的，即将逐步执行上述流程")
            messages.messages_append({"role": "user", "content": "非常好，请按照该流程逐步执行。"})
            is_task_decomposition = False
            is_enhanced_mode = False
            messages = get_chat_response(model=model, 
                                         messages=messages, 
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode, 
                                         delete_some_messages=delete_some_messages, 
                                         is_task_decomposition=is_task_decomposition)
            
       
    if user_input != None:
        if user_input == '1':
            pass
        elif user_input == '2':
            new_user_content = input("好的，输入对模型结果的修改意见：")
            print("好的，正在进行修改。")
            # 在messages中暂时记录上一轮回答的内容
            messages.messages_append(text_answer_message)
            # 记录用户提出的修改意见
            messages.messages_append({"role": "user", "content": new_user_content})

            # 再次调用主函数进行回答，为了节省token，可以删除用户修改意见和第一版模型回答结果
            # 因此这里可以设置delete_some_messages=2
            # 此外，这里需要设置is_task_decomposition=is_task_decomposition
            # 当需要修改复杂任务拆解结果时，会自动带入is_task_decomposition=True
            messages = get_chat_response(model=model, 
                                         messages=messages, 
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode, 
                                         delete_some_messages=2, 
                                         is_task_decomposition=is_task_decomposition)

        elif user_input == '3':
            new_user_content = input("好的，请重新提出问题：")
            # 修改问题
            messages.messages[-1]["content"] = new_user_content
            # 再次调用主函数进行回答
            messages = get_chat_response(model=model, 
                                         messages=messages, 
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode, 
                                         delete_some_messages=delete_some_messages, 
                                         is_task_decomposition=is_task_decomposition)

        else:
            print("好的，已退出当前对话")
        
    # 若不是开发者模式
    else:
        # 记录返回消息
        messages.messages_append(text_answer_message)
    
    return messages

class MateGen():
    def __init__(self, 
                 api_key,
                 model='gpt-3.5-turbo-0613', 
                 system_content_list=[],
                 project=None, 
                 messages=None, 
                 available_functions=None,
                 is_enhanced_mode=False, 
                 is_developer_mode=False):
        """
        初始参数解释：
        api_key：必选参数，表示调用OpenAI模型所必须的字符串密钥，没有默认取值，需要用户提前设置才可使用MateGen；
        model：可选参数，表示当前选择的Chat模型类型，默认为gpt-3.5-turbo-0613，具体当前OpenAI账户可以调用哪些模型，可以参考官网Limit链接：https://platform.openai.com/account/limits ；
        system_content_list：可选参数，表示输入的系统消息或者外部文档，默认为空列表，表示不输入外部文档；
        project：可选参数，表示当前对话所归属的项目名称，需要输入InterProject类对象，用于表示当前对话的本地存储方法，默认为None，表示不进行本地保存；
        messages：可选参数，表示当前对话所继承的Messages，需要是ChatMessages对象、或者是字典所构成的list，默认为None，表示不继承Messages；
        available_functions：可选参数，表示当前对话的外部工具，需要是AvailableFunction对象，默认为None，表示当前对话没有外部函数；
        is_enhanced_mode：可选参数，表示当前对话是否开启增强模式，增强模式下会自动开启复杂任务拆解流程以及深度debug功能，会需要耗费更多的计算时间和金额，不过会换来Agent整体性能提升，默认为False；
        is_developer_mode：可选参数，表示当前对话是否开启开发者模式，在开发者模式下，模型会先和用户确认文本或者代码是否正确，再选择是否进行保存或者执行，对于开发者来说借助开发者模式可以极大程度提升模型可用性，但并不推荐新人使用，默认为False；
        """
        
        self.api_key = api_key
        self.model = model
        self.project = project
        self.system_content_list = system_content_list
        tokens_thr = None
        
        # 计算tokens_thr
        if '1106' in model:
            tokens_thr = 110000
        elif '16k' in model:
            tokens_thr = 12000
        elif '4-0613' in model:
            tokens_thr = 7000
        else:
            tokens_thr = 3000
            
        self.tokens_thr = tokens_thr
        
        # 创建self.messages属性
        self.messages = ChatMessages(system_content_list=system_content_list, 
                                     tokens_thr=tokens_thr)
        
        # 若初始参数messages不为None，则将其加入self.messages中
        if messages != None:
            self.messages.messages_append(messages)
        
        self.available_functions = available_functions
        self.is_enhanced_mode = is_enhanced_mode
        self.is_developer_mode = is_developer_mode
        
    def chat(self, question=None):
        """
        MateGen类主方法，支持单次对话和多轮对话两种模式，当用户没有输入question时开启多轮对话，反之则开启单轮对话。\
        无论开启单论对话或多轮对话，对话结果将会保存在self.messages中，便于下次调用
        """
        head_str = "▌ Model set to %s" % self.model
        display(Markdown(head_str))
        
        if question != None:
            self.messages.messages_append({"role": "user", "content": question})
            self.messages = get_chat_response(model=self.model, 
                                              messages=self.messages, 
                                              available_functions=self.available_functions,
                                              is_developer_mode=self.is_developer_mode,
                                              is_enhanced_mode=self.is_enhanced_mode)
        
        else:
            while True:
                self.messages = get_chat_response(model=self.model, 
                                                  messages=self.messages, 
                                                  available_functions=self.available_functions,
                                                  is_developer_mode=self.is_developer_mode,
                                                  is_enhanced_mode=self.is_enhanced_mode)
                
                user_input = input("您还有其他问题吗？(输入退出以结束对话): ")
                if user_input == "退出":
                    break
                else:
                    self.messages.messages_append({"role": "user", "content": user_input})

    def reset(self):
        """
        重置当前MateGen对象的messages
        """
        self.messages = ChatMessages(system_content_list=self.system_content_list)
    
    def upload_messages(self):
        """
        将当前messages上传至project项目中
        """
        if self.project == None:
            print("需要先输入project参数（需要是一个InterProject对象），才可上传messages")
            return None
        else:
            self.project.append_doc_content(content=self.messages.history_messages)