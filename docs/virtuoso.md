# Azure Virtuoso Image

There is a pay-as-you-go image on MS Azure.
```text
Source image publisher
openlinkswcom-pago

Source image offer
openlink-virtuoso-azure-pago-offer-20201019

Source image plan
openlink-virtuoso-pago-plan-1001
```

###### set password
The default password for `dba` in this image is a part of the instance id which can be accessed using
```shell
curl -H Metadata:true "http://169.254.169.254/metadata/instance/compute/vmId?api-version=2017-08-01&format=text"
# returns xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx 
# the pwd is xxxxxxxx-xxxx-xxxx
```
Now in `isql 1111` do
```sql
set password "xxxxxxxx-xxxx-xxxx" "yyy";
```
to set password to "yyy".

