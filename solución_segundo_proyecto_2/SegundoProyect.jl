### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ 8fa7bbd2-613c-11eb-29f5-8b965431bae5
begin
	using TextAnalysis
	using CSV
	using PlutoUI
end

# ╔═╡ 7bda594a-6144-11eb-0dca-81dd4cbecb8a
begin
	# Carga los datos del CSV
	data = CSV.File("Review_data.csv")
	documents = StringDocument.(data.review_body)
end

# ╔═╡ 00f05ccc-613d-11eb-26e6-e75b58b5eedd
begin
	corpus = Corpus(documents)
	
	# Preprocesamiento del corpus
	remove_case!(corpus)
	prepare!(corpus, strip_non_letters)
	prepare!(corpus, strip_stopwords)
	stem!(corpus)
	
	update_lexicon!(corpus)
	
	# Representación matricial de los documentos
	doc_term_matrix = DocumentTermMatrix(corpus)
	# Generación de una función que convierte de indices a palbras
	index_to_word = Dict(v => k for (k, v) in doc_term_matrix.column_indices)
	get_word = index -> index_to_word[index]
end
	

# ╔═╡ 8428b1fc-613d-11eb-2e94-75b98c2ea94a
"""
	get_topics(ϕ, k)

Extrae las k palabras más probables para cada tema de la matriz ϕ (numtemas x numpalabras)
"""
function get_topics(ϕ, k)
	num_topics, _ = size(ϕ)
	topics = []
	for i in 1:num_topics
		probs = ϕ[i,:]
		indices = partialsortperm(probs, 1:k, rev=true)
		push!(topics, collect(zip(get_word.(indices), probs[indices])))
	end
	return topics
end


# ╔═╡ 33a63174-6146-11eb-2af8-c71f7a41b9ba
"""
	print_topics(topics)

Agarra un lista de temas (lista de tuplas (palabras, probabilidad)) y las imprime
"""
function print_topics(topics)
	for i in 1:length(topics)
		println("Topic $i")
		for (words, probs) in topics[i]
			println("$words\t\t\t$probs")
		end
		println()
	end
end

# ╔═╡ b2606a14-6141-11eb-033e-1fad4ec0bba2
ϕ, θ = lda(doc_term_matrix, 5, 200, 2.0, 0.5)

# ╔═╡ 906f4d66-613f-11eb-2cef-d1d223b5d1ac
begin
	topics = get_topics(ϕ, 15)
	PlutoUI.with_terminal() do
		print_topics(topics)
	end
end

# ╔═╡ Cell order:
# ╠═8fa7bbd2-613c-11eb-29f5-8b965431bae5
# ╠═7bda594a-6144-11eb-0dca-81dd4cbecb8a
# ╠═00f05ccc-613d-11eb-26e6-e75b58b5eedd
# ╠═8428b1fc-613d-11eb-2e94-75b98c2ea94a
# ╠═33a63174-6146-11eb-2af8-c71f7a41b9ba
# ╠═b2606a14-6141-11eb-033e-1fad4ec0bba2
# ╠═906f4d66-613f-11eb-2cef-d1d223b5d1ac
