from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

def get_agg_imps_click(ad_id_list,txns_path,c_path):
    txns = sqlContext.read.json(txns_path)
    c = sqlContext.read.json(c_path)
    txns = txns.withColumn("tzo", txns["tzo"].cast(t.IntegerType()))
    txns = txns.withColumn('usertime', txns['createdtime'] - txns['tzo']*60*1000)
    txns = txns.withColumn("usertime", f.substring("usertime",0,10))
    txns = txns.withColumn("usertime", txns["usertime"].cast("double"))
    txns = txns.withColumn('usertime', f.date_format(txns.usertime.cast(dataType=t.TimestampType()), "yyyy-MM-dd HH:mm:ss.SSS"))
    txns = txns.withColumn('hour', f.hour(txns.usertime.cast('timestamp')))
    txns = txns.withColumn('minute', f.minute(txns.usertime.cast('timestamp')))
    txns = txns.filter(txns.adid.isin(ad_id))
    txns = txns.withColumn('userminute', txns['hour']*60 + txns['minute'])
    txns.createOrReplaceTempView("txns")
    c.createOrReplaceTempView("click")
    merge = spark.sql("SELECT t.adid, t.adrequestid as txns_req, t.rowid, t.hour,t.minute, t.adid, c.adrequestid as click_req FROM txns as t LEFT OUTER JOIN click as c ON t.adrequestid = c.adrequestid where t.userminute between 720 and 900 and t.impgroup = 'TEST' and t.modelsFound = 'YES'")
    merge.createOrReplaceTempView("merge")
    temp = spark.sql("SELECT count(click_req) as clicks,count(*) as imps, rowid, adid  FROM merge group by adid, rowid order by adid")
    temp.write.option("header",True).format("com.databricks.spark.csv").save('/user/shubhamg/cac/mab_testing/dsid_afternoon.csv')
    pandasDF = temp.toPandas()
    return pandasDF



if __name__ == '__main__':
    # bdp transaction and click log path
    txns_path = "/user/cac/ec/prod/T/year=2021/month=06/day=16/hour=*/*transaction.log.*"
    c_path = "/user/cac/ec/prod/C/year=2021/month=06/day=16/hour=*/*transaction_c.log.*"
    ad_id = [6376, 6377, 6378]
    pandasDF = get_agg_imps_click(ad_id_list,txns_path,c_path)