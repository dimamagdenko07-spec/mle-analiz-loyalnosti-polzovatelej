import os
import pandas as pd
from sqlalchemy import create_engine
import psycopg
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import phik

class DataProcessor:
    def __init__(self):
        connection_string = self.__get_env_info()
        self.engine = create_engine(connection_string)

    def get_afisha_df(self):
        query = '''
            SELECT
            	p.user_id,
            	p.device_type_canonical,
            	p.order_id,
            	p.created_dt_msk AS order_dt,
            	p.created_ts_msk AS order_ts,
            	p.currency_code,
            	p.revenue,
            	p.tickets_count,
            	p.created_dt_msk - LAG(p.created_dt_msk) OVER (PARTITION BY user_id ORDER BY p.created_dt_msk) AS days_since_prev,
            	p.event_id,
            	e.event_name_code AS event_name,
            	p.service_name,
            	e.event_type_main,
            	r.region_name,
            	c.city_name
            FROM afisha.purchases p
            LEFT JOIN afisha.events e USING (event_id)
            LEFT JOIN afisha.city c USING (city_id)
            LEFT JOIN afisha.regions r USING (region_id)
            WHERE (p.device_type_canonical IN ('mobile', 'desktop')) AND (e.event_type_main != 'фильм')
            ORDER BY user_id ASC, order_dt ASC
        '''
        df = pd.read_sql_query(query, con=self.engine)
        self.prefiltered_afisha_df = df
        return df

    def get_currency_df(self):
        rub_tenge = pd.read_csv("data/final_tickets_tenge_df.csv")
        rub_tenge.columns = ['order_dt', 'nominal', 'curs', 'cdx']
        rub_tenge['order_dt'] = rub_tenge['order_dt'].astype('datetime64[ns]')
        self.rub_tenge = rub_tenge
        return rub_tenge

    def preprocessing(self, df: pd.DataFrame, currency_df: pd.DataFrame):
        df = df.merge(currency_df, how='left', on='order_dt')
        df['revenue_rub'] = df[df['currency_code'] == 'kzt']['revenue'] / df[df['currency_code'] == 'kzt']['nominal'] * df[df['currency_code'] == 'kzt']['curs']
        df = df[['user_id', 'device_type_canonical', 'order_id', 'order_dt', 'order_ts',
               'currency_code', 'revenue', 'revenue_rub', 'tickets_count', 'days_since_prev',
               'event_id', 'event_name', 'service_name', 'event_type_main',
               'region_name', 'city_name']]
        df['revenue_rub'] = df['revenue_rub'].fillna(df['revenue'])
        df = df.drop_duplicates(subset=['user_id', 'device_type_canonical', 'order_dt', 'order_ts',
       'revenue_rub', 'tickets_count',
       'event_id', 'event_name', 'service_name',
       'event_type_main', 'region_name', 'city_name'])
        df['tickets_count'] = pd.to_numeric(df['tickets_count'], downcast='integer')
        df['event_id'] = pd.to_numeric(df['event_id'], downcast='integer')
        df['order_id'] = pd.to_numeric(df['order_id'], downcast='integer')
        percentile99 = df['revenue_rub'].quantile(0.99)
        df = df[(df['revenue_rub'] < percentile99) & (df['revenue_rub'] > 0)]
        percentile99 = df['tickets_count'].quantile(0.99)
        df = df[df['tickets_count'] < percentile99]
        df = df.reset_index(drop=True)
        self.df = df
        return df

    def create_user_stats_df(self, df = None):
        if df is None and hasattr(self, 'df'):
            df = self.df
        if (not hasattr(self, 'df')) and df is None:
            return None
        user_first_order = df[df['days_since_prev'].isna()][['user_id', 'order_dt', 'device_type_canonical', 'region_name', 'event_type_main']]
        user_first_order.columns = ['user_id', 'first_order_dt', 'device_type_canonical', 'region_name', 'event_type_main']
        user_latest_order = df.groupby('user_id')['order_dt'].max().reset_index()
        user_latest_order.columns = ['user_id', 'last_order_dt']
        user_category_stats = user_first_order.merge(user_latest_order, how='left', on='user_id')
        user_aggregation_stats = df.groupby('user_id').agg({'order_id': 'count', 'revenue_rub' : 'mean', 'days_since_prev' : 'mean'}).reset_index()
        user_aggregation_stats.columns = ['user_id', 'total_orders', 'avg_revenue_rub', 'avg_days_since_prev']
        user_stats = user_category_stats.merge(user_aggregation_stats, how='left', on='user_id')
        user_stats = user_stats[['user_id', 'first_order_dt', 'last_order_dt', 'device_type_canonical', 'region_name', 'event_type_main', 'total_orders', 'avg_revenue_rub', 'avg_days_since_prev']]
        user_stats['is_two'] = user_stats['total_orders'] > 1
        user_stats['is_two'] = user_stats['is_two'].astype('int')
        user_stats = user_stats[user_stats['total_orders'] < user_stats['total_orders'].quantile(0.99)]
        user_stats['avg_days_since_prev_float'] = user_stats['avg_days_since_prev'].dt.total_seconds() / 60 / 60 / 24
        user_stats['first_order_day_of_week'] = user_stats['first_order_dt'].dt.day_name()
        self.user_stats = user_stats
        return user_stats

    def device_segmentation(self, user_stats = None):
        if user_stats is None and (not hasattr(self, 'user_stats')):
            return None
        if user_stats is None:
            user_stats = self.user_stats
        segmentation_device = user_stats.groupby(['device_type_canonical']).agg({'user_id' : 'count', 'is_two' : 'mean'}).reset_index()
        segmentation_device.columns = ['first_device_type', 'total_users', 'loyal_users_ratio']
        segmentation_device["total_users_ratio"] = round(segmentation_device["total_users"] / user_stats.shape[0], 4)
        segmentation_device = segmentation_device.sort_values(by='total_users', ascending=False)
        segmentation_device = segmentation_device[['first_device_type', 'total_users', 'total_users_ratio', 'loyal_users_ratio']]
        return segmentation_device

    def event_segmentation(self, user_stats = None):
        if user_stats is None and (not hasattr(self, 'user_stats')):
            return None
        if user_stats is None:
            user_stats = self.user_stats
        segmentation_event = user_stats.groupby(['event_type_main']).agg({'user_id' : 'count', 'is_two' : 'mean'}).reset_index()
        segmentation_event.columns = ['first_event_type', 'total_users', 'loyal_users_ratio']
        segmentation_event["total_users_ratio"] = round(segmentation_event["total_users"] / user_stats.shape[0], 4)
        segmentation_event = segmentation_event.sort_values(by='total_users', ascending=False)
        segmentation_event = segmentation_event[['first_event_type', 'total_users', 'total_users_ratio', 'loyal_users_ratio']]
        return segmentation_event

    def region_segmentation(self, user_stats = None):
        if user_stats is None and (not hasattr(self, 'user_stats')):
            return None
        if user_stats is None:
            user_stats = self.user_stats
        segmentation_region = user_stats.groupby(['region_name']).agg({'user_id' : 'count', 'is_two' : 'mean'}).reset_index()
        segmentation_region.columns = ['first_region_name', 'total_users', 'loyal_users_ratio']
        segmentation_region["total_users_ratio"] = round(segmentation_region["total_users"] / user_stats.shape[0], 4)
        segmentation_region = segmentation_region.sort_values(by='total_users', ascending=False)
        segmentation_region = segmentation_region[['first_region_name', 'total_users', 'total_users_ratio', 'loyal_users_ratio']]
        return segmentation_region

    def display_device_retention(self, segmentation_device):
        segmentation_device.set_index('first_device_type')['loyal_users_ratio'].plot(kind='bar', rot = 45)
        plt.xlabel("Первое устройство")
        plt.ylabel("Доля лояльных клиентов")
        plt.title("Доля лояльных пользователей по сегментам в разрезе первого устройства")
        plt.show()

    def display_event_retention(self, segmentation_event):
        segmentation_event.set_index('first_event_type')['loyal_users_ratio'].plot(kind='bar', rot = 45)
        plt.xlabel("Первое мероприятие")
        plt.ylabel("Доля лояльных клиентов")
        plt.title("Доля лояльных пользователей по сегментам в разрезе первого мероприятия")
        plt.show()

    def display_region_retention(self, segmentation_region):
        segmentation_region.set_index('first_region_name')['loyal_users_ratio'].head(10).plot(kind='barh')
        plt.xlabel("Регион первого заказа")
        plt.ylabel("Доля лояльных клиентов")
        plt.title("Доля лояльных пользователей по сегментам в разрезе региона первого заказа")
        plt.show()

    def corr_analysis(self, user_stats = None):
        if user_stats is None and (not hasattr(self, 'user_stats')):
            return None
        if user_stats is None:
            user_stats = self.user_stats
        corr_matrix = user_stats[['first_order_dt', 'last_order_dt', 'device_type_canonical',
       'region_name', 'event_type_main', 'total_orders', 'avg_revenue_rub', 'is_two', 'avg_days_since_prev_float',
       'first_order_day_of_week']].phik_matrix(interval_cols=['total_orders', 'avg_revenue_rub', 'avg_days_since_prev_float'])[['total_orders']].drop('total_orders').sort_values(by='total_orders', ascending=False)
        sns.heatmap(data=corr_matrix, annot=True, fmt = '.2f', cmap = 'viridis')
        plt.title("Тепловая карта корреляции столбца total_orders")
        plt.show()

    def __get_env_info(self):
        load_dotenv()
        db_user = os.getenv('DB_USER')
        db_pwd = os.getenv('DB_PWD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')
        db_config = {
            'user' : db_user,
            'pwd' : db_pwd,
            'host' : db_host,
            'port' : db_port,
            'name' : db_name
        }
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            db_config['user'],
            db_config['pwd'],
            db_config['host'],
            db_config['port'],
            db_config['name'],
        )
        return connection_string