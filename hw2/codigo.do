// Métodos avanzados de organización industrial empírica
// Sebastián Chacón
// 20.531.787-2
// [0] Preliminares
clear all
cd "C:\Users\Alumno FEN\Desktop\oi\tarea2"
use autos.dta

// [1] Número de modelos por segmento para cada año. Media por segmento de precio, características y número de modelos

sort year sg5 brand indiv

* Modelos por segmento para cada año
bys year sg5: egen nProductsInSeg = count(indiv)

* Promedio de variables
tabstat price ccw mspeed size nProductsInSeg, by(sg5) stat(mean) f(%9.2f) nototal

// [2] Construcción de variables 

// [a] Variable log(sj/s0)
gen meanUtility = log(sj/s0)

// [b] Variable log(sj/sg1)
gen intraGroupShare = log(sj/sg1)

// [c] Variables instrumentales

* Características de otros productos del mismo segmento
*bys year sg5: egen othersPriceInSeg = total(price)
*replace othersPriceInSeg = othersPriceInSeg - price

bys year sg5: egen othersCcwInSeg = total(ccw)
replace othersCcwInSeg = othersCcwInSeg - ccw

bys year sg5: egen othersMspeedInSeg = total(mspeed)
replace othersMspeedInSeg = othersMspeedInSeg - mspeed

bys year sg5: egen othersSizeInSeg = total(size)
replace othersSizeInSeg = othersSizeInSeg - size

* Características de otros productos de la misma firma en el mismo segmento

*bys year sg5 brand: egen othersPriceInFirm = total(price)
*replace othersPriceInFirm = othersPriceInFirm - ccw

bys year sg5 brand: egen othersCcwInFirm = total(ccw)
replace othersCcwInFirm = othersCcwInFirm - ccw

bys year sg5 brand: egen othersMspeedInFirm = total(mspeed)
replace othersMspeedInFirm = othersMspeedInFirm - mspeed

bys year sg5 brand: egen othersSizeInFirm = total(size)
replace othersSizeInFirm = othersSizeInFirm - size

// [d] Número de productos propios y rivales, en el mismo segmento

bys year sg5 brand: egen nProductsInFirm = count(indiv)
gen nProductsCompetitors = nProductsInSeg - nProductsInFirm

* replace othersPriceInFirm = 0 if nProductsInFirm == 1 
* replace othersCcwInFirm = 0 if nProductsInFirm == 1
* replace othersMspeedInFirm = 0 if nProductsInFirm == 1
* replace othersSizeInFirm = 0 if nProductsInFirm == 1

// Estadistica descriptiva
* Tabla1
tabstat meanUtility intraGroupShare nProductsCompetitors nProductsInFirm, by(sg5) stat(mean sd) f(%9.2f) nototal 
* Tabla2
tabstat othersCcwInSeg othersCcwInFirm othersMspeedInSeg othersMspeedInFirm othersSizeInSeg othersSizeInFirm, by(sg5) stat(mean sd) f(%9.2f) nototal 

// [3] Modelo Logit

// [a] Características y precio
reg meanUtility price ccw mspeed size
outreg2 using tabla3.tex
// [b] Características, precio y dummies por año
reg meanUtility price ccw mspeed size i.year
outreg2 using tabla3.tex, append keep(price ccw mspeed size)
// [c] Características, precio, dummies por año y por firma
reg meanUtility price ccw mspeed size i.year i.brand
outreg2 using tabla3.tex, append keep(price ccw mspeed size)

// [4] Modelo Logit con IV

// [a] 
ivregress 2sls meanUtility (price = othersCcwInSeg othersMspeedInSeg othersSizeInSeg nProductsCompetitors) ccw mspeed size i.year i.brand
outreg2 using tabla4.tex, keep(price ccw mspeed size)

// [b]
ivregress 2sls meanUtility (price = othersCcwInSeg othersCcwInFirm othersMspeedInSeg othersMspeedInFirm othersSizeInSeg othersSizeInFirm nProductsCompetitors nProductsInFirm) ccw mspeed size i.year i.brand
outreg2 using tabla4.tex, append keep(price ccw mspeed size)


// [5] Modelo Nested Logit
ivregress 2sls meanUtility (price = othersPriceInSeg othersPriceInFirm othersCcwInSeg othersCcwInFirm othersMspeedInSeg othersMspeedInFirm othersSizeInSeg othersSizeInFirm nProductsCompetitors nProductsInFirm) intraGroupShare ccw mspeed size i.year i.brand

// [6]
// [7]








