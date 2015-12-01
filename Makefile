NOVEL = novel

novel: $(NOVEL)/novel.pdf
	
$(NOVEL)/novel.pdf: $(NOVEL)/99/map.tex
	cp frame.tex $(NOVEL)/novel.tex
	cd $(NOVEL);\
		latexmk --pdf novel.tex

$(NOVEL)/99/map.tex: $(NOVEL)/99/map.png
	for dir in $(NOVEL)/??; do \
		cd $$dir; \
		pandoc -t latex --chapters map.md -o map.tex; \
		cd -; \
	done

$(NOVEL)/99/map.png:
	python generate.py $(NOVEL)

