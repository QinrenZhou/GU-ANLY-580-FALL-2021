# Lab 06: Query matching with FastText and Virtex


## Launch a FastText service using the Virtex library in Python
The purpose of this lab is to expose you to a few things:

(a) The [FastText](https://fasttext.cc/docs/en/python-module.html) library in Python, which is a language modeling and text classification framework

(b) The idea that NLP systems typically take on the form of microservices, wherein specific functions, such as computing embeddings or computing similar words, are performed in isolation and accessed through HTTP requests (or other protocols such as gRPC).

(c) The [Advanced Rest Client](https://install.advancedrestclient.com/install) (ARC), which allows you to make HTTP requests containing data (such as JSON) to an HTTP endpoint.

(d) The [Virtex](https://pypi.org/project/virtex/) library, which provides a convenient way to expose your machine learning computation as a service over HTTP without having to write any networking code (see `query_matching_demo.py`, `query_matching_demo.sh`).


## Task I (20 pts)

1. Launch the FastText query matching service by running the following command from the terminal from within the `labs/lab-06/` directory:

    $ ./query_matching_demo.sh

2. Open your Advanced Rest Client application (you need to download it first)

    a. Enter `http://0.0.0.0:580` into the Request URL bar
    
    b. In the Body content type field choose `application/json`

    c. Click the body tab and enter `{"data": ["dogs", "bakery", "hose", "florida", "supreme"]}`

    d. Explore FastText by changing the words and looking at the matches. Paste of a few of the responses below:

    ``` 
    {"data": ["dogs", "bakery", "hose", "florida", "supreme"]}:
    Response: {"data":[[[0.9300785064697266,"arctiidae"],[0.9237712621688843,"noctuidae"],[0.9216641783714294,"eupithecia"],[0.9213652610778809,"mollusk"],[0.9206809401512146,"geometridae"],[0.9193852543830872,"burrows"],[0.9190572500228882,"homo"],[0.9189303517341614,"hunting"],[0.9168787002563477,"tortricidae"],[0.9140787720680237,"conus"]],[[0.9736358523368835,"specialty"],[0.9631682634353638,"slogan"],[0.9589007496833801,"tavern"],[0.9555143713951111,"restaurant"],[0.9517912864685059,"steakhouse"],[0.9470339417457581,"flavors"],[0.9431126713752747,"adobe"],[0.9428221583366394,"corner"],[0.942036509513855,"1860s"],[0.9348995685577393,"missouri"]],[[0.0,"in"],[0.0,"."],[0.0,"of"],[0.0,"and"],[0.0,"a"],[0.0,"</s>"],[0.0,"("],[0.0,"zeynally"],[0.0,"徐嬌"],[0.0,"romanizations"]],[[0.9168912172317505,"decree"],[0.8954152464866638,"jurisdiction"],[0.8942437171936035,"salisbury"],[0.8920913338661194,"opportunities"],[0.8886478543281555,"biotechnology"],[0.8827422261238098,"depending"],[0.8791956901550293,"accreditation"],[0.8787104487419128,"collegiate"],[0.8769832253456116,"graduates"],[0.8687768578529358,"inns"]],[[0.9724624752998352,"labor"],[0.9721168279647827,"officer"],[0.9707459211349487,"commissioner"],[0.9703463315963745,"constituency"],[0.9677392840385437,"servant"],[0.9677126407623291,"regent"],[0.9675344824790955,"mohammed"],[0.9650437831878662,"tenure"],[0.9646663665771484,"tax"],[0.9618818163871765,"brigadier"]]],"error":null}

    {"data": ["meachine", "learning"]}
    {"data":[[[0.0,"in"],[0.0,"."],[0.0,"of"],[0.0,"and"],[0.0,"a"],[0.0,"</s>"],[0.0,"("],[0.0,"zeynally"],[0.0,"徐嬌"],[0.0,"romanizations"]],[[0.9775598049163818,"students"],[0.9679409265518188,"pharmacy"],[0.9582656025886536,"accreditation"],[0.9504334926605225,"accredited"],[0.9497060775756836,"universities"],[0.9487293362617493,"study"],[0.9467121362686157,"campus"],[0.943805456161499,"sma"],[0.9418812990188599,"theology"],[0.9407311677932739,"thane"]]],"error":null}

    {"data": ["today", "is","a","good","day"]}
    {"data":[[[0.9484300017356873,"week"],[0.9435609579086304,"dylan"],[0.9415485858917236,"psychology"],[0.9336086511611938,"kabuki"],[0.9329304099082947,"capitalism"],[0.9318807721138,"linguistics"],[0.9308383464813232,"verse"],[0.9290257096290588,"consciousness"],[0.9288349151611328,"esperanto"],[0.9266324043273926,"prisoner"]],[[0.5923686623573303,"small"],[0.5679067373275757,"extends"],[0.5599420070648193,"locality"],[0.5559906363487244,"north-central"],[0.5538623332977295,"code"],[0.5505450963973999,"southern"],[0.5499141812324524,"southwestern"],[0.5422728657722473,"boating"],[0.5402816534042358,"south-central"],[0.5331577062606812,"thrissur"]],[[0.7105932831764221,"no"],[0.6952199935913086,"per"],[0.6716886162757874,"1998"],[0.6492639183998108,"costs"],[0.644737720489502,"52"],[0.6431398987770081,"life"],[0.6378374099731445,"series"],[0.636232852935791,"full"],[0.6276789903640747,"stationed"],[0.6238977909088135,"bound"]],[[0.8502567410469055,"contained"],[0.8406635522842407,"contain"],[0.8380084037780762,"contains"],[0.8067858219146729,"avengers"],[0.8022655248641968,"tie-in"],[0.7980409264564514,"sin"],[0.7872011065483093,"morality"],[0.7839263677597046,"eleventh"],[0.7833080887794495,"brussels"],[0.781639039516449,"another"]],[[0.9170036911964417,"theatrical"],[0.8809421062469482,"dramatic"],[0.8522061705589294,"treasure"],[0.8441754579544067,"troubled"],[0.8367208242416382,"conflict"],[0.8348316550254822,"young"],[0.8234155774116516,"mr"],[0.8220604658126831,"factor"],[0.8215208649635315,"drama"],[0.8182297348976135,"billie"]]],"error":null}


    {"data": ["the","US","and","European","stock","markets","closed","down","collectively","on","Friday"]}
    {"data":[[[0.7701268792152405,"on"],[0.609031617641449,"la"],[0.6080572605133057,"les"],[0.604745090007782,"of"],[0.6032161116600037,"war"],[0.5898455381393433,"life"],[0.5737563967704773,"our"],[0.5712148547172546,"passion"],[0.5709637999534607,"hero"],[0.5645501613616943,"i"]],[[0.9530154466629028,"polish"],[0.937946617603302,"hal"],[0.9312207102775574,"ve"],[0.9309667348861694,"victim"],[0.9215530157089233,"gale"],[0.9100198149681091,"nash"],[0.8899438977241516,"experiments"],[0.8892269730567932,"displayed"],[0.885275661945343,"aids"],[0.8816126585006714,"ending"]],[[0.6980509161949158,"period"],[0.6786696314811707,"nigel"],[0.6657900214195251,"von"],[0.6620246171951294,"enrique"],[0.6522108316421509,"hong"],[0.6480748653411865,"moore"],[0.6447950005531311,"morel"],[0.643119215965271,"dane"],[0.6406552791595459,"bing"],[0.6295439004898071,"boyd"]],[[0.8147222399711609,"head"],[0.8088144659996033,"zealand"],[0.8014913201332092,"body"],[0.7630527019500732,"defined"],[0.7624486088752747,"existing"],[0.7582453489303589,"tip"],[0.7578800916671753,"2000s"],[0.7532284259796143,"caused"],[0.7506780624389648,"arabian"],[0.7500506043434143,"mainland"]],[[0.9458131194114685,"line"],[0.9199408292770386,"high-performance"],[0.903081476688385,"bulk"],[0.8872323632240295,"nantucket"],[0.8864998817443848,"gauge"],[0.8809571266174316,"liquid"],[0.8758209943771362,"types"],[0.8586945533752441,"railway"],[0.8567088842391968,"50"],[0.8516448140144348,"guides"]],[[0.9440401792526245,"kits"],[0.8963531255722046,"engaged"],[0.893185555934906,"mechanical"],[0.8830302357673645,"express"],[0.8739184737205505,"operators"],[0.8701598644256592,"option"],[0.8655140995979309,"1940s"],[0.8635396957397461,"automobile"],[0.8590028285980225,"wellington"],[0.858173668384552,"zero"]],[[0.9195520281791687,"aqueduct"],[0.9062976837158203,"westward"],[0.892254114151001,"fork"],[0.8922039270401001,"internment"],[0.8848778605461121,"calico"],[0.8847863078117371,"adjacent"],[0.8753249645233154,"highpoint"],[0.8739216923713684,"upstream"],[0.8729236721992493,"lincolnshire"],[0.8715097904205322,"acre"]],[[0.9407320022583008,"weather"],[0.9115602374076843,"planets"],[0.9078132510185242,"dimensions"],[0.9047996401786804,"survey"],[0.8882930278778076,"viking"],[0.8848301768302917,"spirit"],[0.8839735388755798,"discontinued"],[0.8826397657394409,"ns"],[0.8815337419509888,"tipo"],[0.8771337866783142,"solar"]],[[0.9326622486114502,"topic"],[0.9195308089256287,"com"],[0.9122110605239868,"shadows"],[0.9107563495635986,"dream"],[0.9076220393180847,"mega"],[0.905280590057373,"matchday"],[0.9049325585365295,"marvel"],[0.902069628238678,"universe"],[0.9019912481307983,"sundays"],[0.9018844366073608,"ongoing"]],[[0.7721160650253296,"war"],[0.7701266407966614,"the"],[0.767285168170929,"life"],[0.7570273280143738,"of"],[0.750281035900116,"beyond"],[0.7301539778709412,"s"],[0.7266355752944946,"our"],[0.7223485708236694,"go"],[0.7081746459007263,"i"],[0.7060943245887756,"friends"]],[[0.930455207824707,"knows"],[0.9302462935447693,"trip"],[0.9224633574485779,"dawn"],[0.9163205027580261,"forgotten"],[0.9068953394889832,"der"],[0.9045732617378235,"echo"],[0.8981975317001343,"friend"],[0.8979840874671936,"barwick"],[0.8952948451042175,"story"],[0.8951480388641357,"in-depth"]]],"error":null}

    ``` 

**Note for Windows users**: Virtex does not run on Windows (because it requires uvloop which unfortunately does not support Windows). As an alternative, you can run this on your machine using [Docker](https://docs.docker.com/desktop/windows/install/). Once you have docker installed, execute the following commands from within the `GU-ANLY-580/labs/lab-06` folder:

    $ docker build -t fasttext-demo .
    $ docker run -p 580:580 fasttext-demo:latest

And then head over to your rest client (step 2 above) to complete the task. Alternatively, you can interact with FastText programmatically from within Python, for example:

   ```python
   import fasttext
   
   model = fasttext.load_model("en.quant.bin")
   match = model.get_nearest_neighbors("italian")
   ```
