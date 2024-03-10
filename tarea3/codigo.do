// Métodos avanzados de organización industrial empírica
// Sebastián Chacón
// 20.531.787-2
// [0] Preliminares
clear all
global main "C:\Users\Alumno FEN\Desktop\oi\tarea3"
global data "$main/Datos"
global fig "$main/Graficos"
global tablas "$main/Tablas"
cd "$data"
use cardata.dta
* ssc install blp

// [1] Estimación de demanda BLP

* Variables relevantes
bys year: egen totalShares = total(market_share) // suma total de participaciones por año
gen s0 = 1 - totalShares // participacion de outside option
gen meanUtility = log(market_share/s0) // utilidad promedio
gen logPrice = log(price)

* Variables instrumentales en la misma firma
bys year firmid: egen othersHPWeight_1 = total(horsepower_weight)
replace othersHPWeight_1 = othersHPWeight_1 - horsepower_weight

bys year firmid: egen othersLengthWidth_1 = total(length_width)
replace othersLengthWidth_1 = othersLengthWidth_1 - length_width

bys year firmid: egen othersMP_1 = total(miles_per_dollar)
replace othersMP_1 = othersMP_1 - miles_per_dollar

* Variables instrumentales en el mismo año
bys year: egen othersHPWeight_2 = total(horsepower_weight)
replace othersHPWeight_2 = othersHPWeight_2 - horsepower_weight

bys year: egen othersLengthWidth_2 = total(length_width)
replace othersLengthWidth_2 = othersLengthWidth_2 - length_width

bys year: egen othersMP_2 = total(miles_per_dollar)
replace othersMP_2 = othersMP_2 - miles_per_dollar

// [1.1.] Replicar tabla III

* Modelo Logit
reg meanUtility horsepower_weight ac_standard miles_per_dollar length_width price
cd "$tablas"
outreg2 using tabla1.tex
* Modelo Logit VI
ivregress 2sls meanUtility (price = othersHPWeight_1 othersLengthWidth_1 othersMP_1 othersHPWeight_2 othersLengthWidth_2 othersMP_2) horsepower_weight ac_standard miles_per_dollar length_width
outreg2 using tabla1.tex, append 

* Modelo de oferta simple
gen logHP = log(horsepower_weight)
gen logSize = log(length_width)
gen logMP = log(miles_per_dollar)

reg logPrice logHP ac_standard logMP logSize
outreg2 using tabla2.tex 

// [1.2.] Replicar tabla IV
gen constant = 1
gen othersHPWeight3 = othersHPWeight_2 - othersHPWeight_1
gen othersLengthWidth_3 = othersLengthWidth_2 - othersLengthWidth_1
gen othersMP_3= othersMP_2 - othersMP_1
blp market_share constant horsepower_weight ac_standard miles_per_dollar length_width, stochastic(constant price horsepower_weight ac_standard miles_per_dollar length_width) endog(price = othersHPWeight_1 othersLengthWidth_1 othersMP_1 othersHPWeight_2 othersLengthWidth_2 othersMP_2 othersHPWeight3 othersLengthWidth_3 othersMP_3) markets(year)



// [2] Estimación de funciones de producción (OP,LP)
clear all
cd "$data"
use ChileAnalysis.dta
* ssc install prodest
* ssc install st0145_2 

// [2.1.] Variables en logaritmo natural
keep id year va routput age K L M N S V rinvest cexit ciiu_3d
gen logva = log(va)
gen logroutput = log(routput)
gen logkapital = log(K)
gen loglabor = log(L)
gen logmaterial = log(M)
gen logenergy = log(N)
gen logserv = log(S)
gen logv = log(V)
gen logrinvest = log(rinvest)

gen age2 = age^2

// [2.2.] Estimación OP - Inversión como proxy
xtset id year
save data.dta

opreg logroutput, exit(cexit) state(age age2 logkapital) proxy(logrinvest) free(logenergy logmaterial loglabor) second vce(bootstrap, seed(1) rep(50))

// [2.3.] Estimación OP - Inversión como proxy - Industria textil
keep if ciiu_3d == 321
save textil.dta

opreg logroutput, exit(cexit) state(age age2 logkapital) proxy(logrinvest) free(logenergy logmaterial loglabor) cvars (year) second vce(bootstrap, seed(1) rep(50))

// [2.4.] Estimación OP - Materiales como proxy
clear
use data
opreg logroutput, exit(cexit) state(age age2 logkapital) proxy(logmaterial) free(logenergy loglabor) cvars (year) second vce(bootstrap, seed(1) rep(50))

// [2.5.] Estimación OP - Materiales como proxy - Industria textil
clear
use textil
opreg logroutput, exit(cexit) state(age age2 logkapital) proxy(logmaterial) free(logenergy loglabor) cvars (year) second vce(bootstrap, seed(1) rep(50))

// [2.6.] Comparación

// [2.7.] Estimación LP - Materiales como proxy
clear
use data
levpet logroutput, free(logenergy loglabor age age2) proxy(logmaterial) capital(logkapital) revenue i(id) t(year) reps(50)
predict productivity, omega

// [2.8.] Estimación LP - Industria textil - Materiales como proxy
clear
use textil
levpet logroutput, free(logenergy loglabor age age2) proxy(logmaterial) capital(logkapital) revenue i(id) t(year) reps(50)
predict productivity, omega
gen year_aux = year-1900
graph box productivity, over(year_aux) title(Evolución de la productividad) ytitle("")
// [2.9.] Estimación OP - Variable dependiente VA - Industria textil - Materiales como proxy
clear
use textil
opreg logva, exit(cexit) state(age age2 logkapital) proxy(logmaterial) free(logenergy loglabor) cvars (year) second vce(bootstrap, seed(1) rep(50))

// [2.10.] Estimación LP - Variable dependiente VA - Industria textil - Materiales como proxy
clear
use textil
levpet logva, free(logenergy loglabor age age2) proxy(logmaterial) capital(logkapital) valueadded i(id) t(year) reps(50)

// [2.11.] Comparación
