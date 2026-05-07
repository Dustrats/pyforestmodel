/* --Generate SAS librefs for caslibs-- */

/* Create a default CAS session and create SAS librefs for existing caslibs */
/* so that they are visible in the SAS Studio Libraries tree. */
cas mySession;
caslib _all_ assign;


proc casutil;
    load data=parquet.employees_raw
    outcaslib="casuser"
    casout="employee_churn_pqk";
quit;

proc casutil;
   promote casdata="employee_churn_pqk" 
           incaslib="CASUSER" 
           outcaslib="CASUSER";
quit;