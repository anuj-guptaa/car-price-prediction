FROM python:3.11.4-bookworm

RUN pip3 install ipykernel

RUN pip3 install numpy===1.25.2
RUN pip3 install pandas==2.1.0
RUN pip3 install scikit-learn==1.3.0
RUN pip3 install matplotlib
RUN pip3 install dash
RUN pip3 install seaborn
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]

CMD tail -f /dev/null
