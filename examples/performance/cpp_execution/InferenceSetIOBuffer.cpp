//-----------------------------------------------------------------------------
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//-----------------------------------------------------------------------------

#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "/opt/qti-aic/dev/inc/QAicApi.hpp"
#include "/opt/qti-aic/dev/inc/qaicapihpp/QAicApiDataTypes.hpp"

#define WARN_IF(condition, message) \
    do { \
        if (condition) { \
            std::cerr << "Warning: " << message << std::endl; \
        } \
    } while (0)

namespace py = pybind11;

namespace
{

    /**
     * Simple helper to return true if the buffer mapping instance is an input one
     * @param bufmap buffer mapping instance
     * @return true if the instance is an input buffer one.
     */
    [[nodiscard]] bool isInputBuffer(const qaic::rt::BufferMapping &bufmap)
    {
        return bufmap.ioType == BUFFER_IO_TYPE_INPUT;
    }

    class QBufferWrapper
    {
    public:
        explicit QBufferWrapper(size_t size) : buffer_{size, new uint8_t[size]} {}
        ~QBufferWrapper() { delete[] buffer_.buf; }

        [[nodiscard]] QBuffer &getQBuffer() { return buffer_; }

    private:
        QBuffer buffer_;
    };
    using shQBufferWrapper = std::shared_ptr<QBufferWrapper>;

    [[nodiscard]] shQBufferWrapper
    createBuffer(const std::string &bufferName,
                 const qaic::rt::BufferMappings &allBufferMappings,
                 bool isDecode)
    {
        auto it =
            std::find_if(allBufferMappings.begin(), allBufferMappings.end(),
                         [&bufferName](const qaic::rt::BufferMapping &bufferMapping)
                         {
                             return (bufferName == bufferMapping.bufferName);
                         });
        if (it != allBufferMappings.end() and !isDecode)
        {
            return std::make_shared<QBufferWrapper>(it->size);
        }
        else if (it != allBufferMappings.end() and isDecode)
        {
            return std::make_shared<QBufferWrapper>(1);
        }

        throw std::runtime_error(
            "Buffer mapping of Input Type not found for buffer named : " +
            bufferName);
    }

    /**
     * Consuming output from output Buffers into token vector
     * @param outputBuffers Vector to use in case this is an output instance
     * @param logits Vector to store output["logits"].argmax
     * @param generated_ids Vector to store generated ids
     * @param batch_size Batch size
     * @param size_of_logits Total tokens in 1 batch
     */
    void get_logits_from_output_buffers(
    std::vector<QBuffer> &outputBuffers,
    std::vector<std::vector<int64_t>> &logits,
    std::vector<std::vector<int64_t>> &generated_ids,
    int batch_size,
    int size_of_logits)
    {
        for (int i = 0; i < batch_size; ++i) {
            auto rawOPBufPtr = outputBuffers.back().buf + (size_of_logits * sizeof(float) * i);
            const float *buffer = reinterpret_cast<const float *>(rawOPBufPtr);
            auto maxElementIter = std::max_element(buffer, buffer + size_of_logits);
            int maxElementIndex = std::distance(buffer, maxElementIter);

            logits[i] = {maxElementIndex};
            generated_ids[i].push_back(maxElementIndex);
        }
    }

    /**
     * Given a Input buffer and size, populate it with inputs data(input_ids or position ids)
     * @param inputBuffer buffer to populate
     * @param tokenVector vector to fill input Buffer
     */
    void populateBuffer(QBuffer &inputBuffer,
                        const std::vector<std::vector<int64_t>> &tokenVector)
    {
        size_t token_size_bytes = (tokenVector.size() * tokenVector[0].size() * sizeof(int64_t));
        if (inputBuffer.size < token_size_bytes)
        {
            delete[] inputBuffer.buf;
            inputBuffer.buf = new uint8_t[token_size_bytes];
            inputBuffer.size = token_size_bytes;
        }

        auto startPtr = inputBuffer.buf;
        for (const auto& row : tokenVector)
        {
            std::copy_n(reinterpret_cast<const uint8_t *>(row.data()),
                        (row.size() * sizeof(int64_t)),
                        inputBuffer.buf);
            inputBuffer.buf += row.size() * sizeof(int64_t);
        }
        inputBuffer.buf = startPtr;
    }

    // template <typename T>
    // [[nodiscard]] std::string qBufferToString(shQBufferWrapper wrappedBuf)
    // {
    //     std::ostringstream strm;
    //     auto rawBufPtr = wrappedBuf->getQBuffer().buf;
    //     const T *bufferT = reinterpret_cast<const T *>(rawBufPtr);
    //     int numT = wrappedBuf->getQBuffer().size / sizeof(T);
    //     for (int i = 0; i < numT; i++)
    //     {
    //         strm << "[ " << i << " ] = " << bufferT[i] << "\n";
    //     }
    //     return strm.str();
    // }

    /**
     * Given buffer mapping instance, return true if this instance does not
     * contain input or output buffers (e.g. it contains uninitialized or invalid)
     * @param bufmap buffer mapping instance
     * @return true if the buffer mapping instance does not container a valid buffer
     */
    [[nodiscard]] bool notInputOrOutput(const qaic::rt::BufferMapping &bufmap)
    {
        const std::initializer_list<QAicBufferIoTypeEnum> bufTypes{
            BUFFER_IO_TYPE_INPUT, BUFFER_IO_TYPE_OUTPUT};
        const auto func([type = bufmap.ioType](const auto v)
                        { return v == type; });
        return std::none_of(bufTypes.begin(), bufTypes.end(), func);
    }

    /**
     * Given input and output buffers, release all heap allocated
     * @param bufferMappings vector of BufferMapping
     * @param inputBuffers vector of QBuffers - inputs
     * @param outputBuffers vector of Qbuffers - outputs
     * @param inputIdBuffers Qbuffers - input id
     * @param positionIdBuffers Qbuffers - position id
     */
    void populateBuffersWithInputs(const std::vector<qaic::rt::BufferMapping> bufferMappings,
                                   std::vector<QBuffer> &inputBuffers,
                                   std::vector<QBuffer> &outputBuffers,
                                   QBuffer &inputIdBuffer,
                                   QBuffer &positionIdBuffer)
    {
        inputBuffers.clear();
        outputBuffers.clear();
        for (const auto &bufmap : bufferMappings)
        {
            QBuffer buf{bufmap.size, new uint8_t[bufmap.size]};
            if (notInputOrOutput(bufmap))
            {
                continue;
            }
            else if (isInputBuffer(bufmap))
            {
                inputBuffers.push_back(buf);
            }
            else
            {
                outputBuffers.push_back(buf);
            }
        }

        // Filling last 2 index of inputBuffers with inputIds and positionIds
        inputBuffers[inputBuffers.size() - 1] = positionIdBuffer;
        inputBuffers[inputBuffers.size() - 2] = inputIdBuffer;
    }
} // namespace

int generatePrompt(
    py::object tokenizer,
    const std::string &qpcPath,
    int prompt_len,
    int ctx_len,
    int batch_size,
    std::optional<std::vector<std::string>> prompt = std::nullopt,
    std::optional<int> generation_len = std::nullopt,
    std::optional<std::vector<int>> device_id = std::nullopt)
{
    try
    {
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("append")("examples/cpp_execution");
        py::module text_generation_inference = py::module::import("text_inference_using_cpp");

        // QID Generation
        std::vector<QID> qidList;
        if (device_id.has_value())
        {
            for (const auto &id : device_id.value())
            {
                try
                {
                    int32_t qid = id;
                    qidList.push_back(qid);
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Invalid device id string" << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "Device id string " << id << " is out of range!" << std::endl;
                }
            }
        }
        else
        {
            // need to use auto device picker
            qidList.push_back(0);
        }
        // *** CONTEXT ***
        constexpr QAicContextProperties_t *NullProp = nullptr;
        auto context = qaic::rt::Context::Factory(NullProp, qidList); // session == context

        // *** QPC ***
        auto qpc = qaic::rt::Qpc::Factory(qpcPath);

        // TODO: prefill_seq_len  from context
        int prefill_seq_len = prompt_len;
        py::dict inputs = text_generation_inference.attr("tokenize_for_prefill")(prompt, tokenizer).cast<py::dict>();

        // Creating attention_mask
        py::array attention_mask_py = inputs["attention_mask"].cast<py::array>();
        auto attn_mask_buff = attention_mask_py.request();
        int64_t *attn_mask_ptr = static_cast<int64_t *>(attn_mask_buff.ptr);
        std::vector<int64_t> attention_mask_sum; //Equal to position_ids in python
        for (ssize_t i = 0; i < attn_mask_buff.shape[0]; ++i)
        {
            int axis_1_sum = 0;
            for (ssize_t j = 0; j < attn_mask_buff.shape[1]; j++)
            {
                axis_1_sum += attn_mask_ptr[i * (attn_mask_buff.shape[1]) + j];
            }

            attention_mask_sum.push_back(axis_1_sum);
        }

        py::array input_ids_array = inputs["input_ids"].cast<py::array>();
        auto input_ids_array_buff = input_ids_array.request();
        ssize_t padded_len = input_ids_array_buff.shape[1];
        int num_chunks = static_cast<int>(std::ceil(static_cast<double>(padded_len) / prefill_seq_len));
        padded_len = num_chunks * prefill_seq_len; // Convert to a multiple of prompt_len

        if (generation_len.has_value() and generation_len.value() <= 0)
        {
            throw std::runtime_error("Error: Generation Len is <= 0");
        }

        // Calculate the max generation length.
        int max_gen_len = ctx_len - *(std::max_element(attention_mask_sum.begin(), attention_mask_sum.end()));;
        if (!generation_len.has_value())
        {
            generation_len = max_gen_len;
        }
        WARN_IF(generation_len.value() > max_gen_len, "Passed generation_len is greater than allowed length.");

        // Getting inputs dict from Python
        inputs = text_generation_inference.attr("tokenize_for_prefill_with_padded_len")(prompt, tokenizer, padded_len).cast<py::dict>();

        // PREPARE INPUTS FOR PREFILL
        std::vector<u_int64_t> arrange_vector(padded_len);
        for (int i = 0; i < (int)arrange_vector.size(); ++i)
        {
            arrange_vector[i] = i;
        }

        // Create position_ids vector
        std::vector<std::vector<int64_t>> position_ids;
        for(ssize_t i = 0; i < attn_mask_buff.shape[0]; ++i)
        {
            std::vector<int64_t> position_ids_value;
            for (int64_t j = 0; j< padded_len; ++j)
            {
                if((j < attn_mask_buff.shape[1]) && attn_mask_ptr[i * (attn_mask_buff.shape[1]) + j] == 1)
                {
                    position_ids_value.push_back(arrange_vector[j]);
                }
                else
                {
                    position_ids_value.push_back(-1);
                }
            }
            position_ids.push_back(position_ids_value);
            position_ids_value.clear();
        }

        // Create input_ids vector
        py::array input_ids_py = inputs["input_ids"].cast<py::array>();
        py::buffer_info inp_id_buf = input_ids_py.request();
        std::vector<std::vector<int64_t>> input_ids;
        int64_t *input_id_ptr = static_cast<int64_t *>(inp_id_buf.ptr);
        for (ssize_t i = 0; i < inp_id_buf.shape[0]; ++i)
        {
            std::vector<int64_t> input_ids_value;
            for (ssize_t j = 0; j < inp_id_buf.shape[1]; ++j)
            {
                input_ids_value.push_back(input_id_ptr[i * (inp_id_buf.shape[1]) + j]);
            }
            input_ids.push_back(input_ids_value);
            input_ids_value.clear();
        }

        // Max value from position ids 2D vector for all batches
        std::vector<int64_t> max_input_len_value_array;
        for(int i=0;i<batch_size;i++)
        {
            auto max_input_len_value = std::max_element(position_ids[i].begin(), position_ids[i].end());
            max_input_len_value_array.push_back(*max_input_len_value);
        }

        // *** INFERENCE SET CREATION ***
        constexpr uint32_t setSize = 10;
        constexpr uint32_t numActivations = 1;
        auto inferenceSet = qaic::rt::InferenceSet::Factory(
            context, qpc, qidList.at(0), setSize, numActivations);

        // *** SETUP IO BUFFERS ***
        qaic::rt::shInferenceHandle submitHandle;
        auto status = inferenceSet->getAvailable(submitHandle);
        if (status != QS_SUCCESS)
        {
            std::cerr << "Error obtaining Inference Handle\n";
            return -1;
        }

        constexpr uint32_t inferenceId = 0; // also named as request ID
        qaic::rt::shInferenceHandle completedHandle;

        // Making _past values as NULL
        const auto &bufferMappings = qpc->getBufferMappings();
        const auto &bufferMappings2 = qpc->getBufferMappingsV2();

        qaic::rt::BufferIdentifiers bufferIdentifiers(bufferMappings2);
        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> bufDim = bufferIdentifiers.getBufferSizeDimensionPair();

        for (auto &bufid : bufferIdentifiers.getBufferIdentifierVec())
        {
            if (bufid.getBufferName().find("past_") == 0)
            {
                bufDim[bufid.getBufferIndex()].second = std::vector{0U};
            }
        }
        submitHandle->setBufferDimensions(bufDim);

        // *** BUFFER CREATION ***
        auto inputIdBuffer = createBuffer("input_ids", bufferMappings, false);
        auto positionIdBuffer = createBuffer("position_ids", bufferMappings, false);
        std::vector<QBuffer> inputBuffers;
        std::vector<QBuffer> outputBuffers;

        //*** RUN PREFILL ***
        auto startPrefill = std::chrono::system_clock::now();

        for(int i=0;i < num_chunks; i++)
        {
            //*** CHUNKING ***
            std::vector<std::vector<int64_t>> sliced_input_ids;
            for(int j=0;j<(int)input_ids.size();j++)
            {
                std::vector<int64_t> sliced_input_ids_value(input_ids[j].begin() + i * prefill_seq_len,
                                                        input_ids[j].begin() + (i + 1) * prefill_seq_len);
                sliced_input_ids.push_back(sliced_input_ids_value);
            }

            std::vector<std::vector<int64_t>> sliced_position_ids;
            for(int j=0;j<(int)position_ids.size();j++)
            {
                std::vector<int64_t> sliced_position_ids_value(position_ids[j].begin() + i * prefill_seq_len,
                                                        position_ids[j].begin() + (i + 1) * prefill_seq_len);
                sliced_position_ids.push_back(sliced_position_ids_value);
            }

            // *** POPULATE BUFFERS ***
            populateBuffer(inputIdBuffer->getQBuffer(), sliced_input_ids);
            populateBuffer(positionIdBuffer->getQBuffer(), sliced_position_ids);

            populateBuffersWithInputs(bufferMappings,
                                    inputBuffers,
                                    outputBuffers,
                                    inputIdBuffer->getQBuffer(),
                                    positionIdBuffer->getQBuffer());

            // *** SET BUFFERS ***
            submitHandle->setInputBuffers(inputBuffers);
            submitHandle->setOutputBuffers(outputBuffers);

            // *** SUBMIT ***
            status = inferenceSet->submit(submitHandle, inferenceId);
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in submitting handle through InferenceSet\n";
                return -1;
            }

            // *** COMPLETION ***
            status = inferenceSet->getCompletedId(completedHandle, inferenceId);
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in getting completed handle through InferenceSet\n";
                return -1;
            }
            status = inferenceSet->putCompleted(std::move(completedHandle));
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in putting completed handle through InferenceSet\n";
                return -1;
            }
        }

        auto prefillEnd = std::chrono::high_resolution_clock::now();
        // *** GET OUTPUT ***
        //
        // At this point, the output is available in "outputBuffers" and can be
        // consumed.

        // *** BUFFER DIMS UPDATE FOR DECODE***
        for (auto &bufid : bufferIdentifiers.getBufferIdentifierVec())
        {
            if (bufid.getBufferName().find("input_ids") == 0)
            {
                int size = bufDim[bufid.getBufferIndex()].second.size();
                bufDim[bufid.getBufferIndex()].first = sizeof(int64_t);
                bufDim[bufid.getBufferIndex()].second[size-1] = 1;
            }
            if (bufid.getBufferName().find("position_ids") == 0)
            {
                int size = bufDim[bufid.getBufferIndex()].second.size();
                bufDim[bufid.getBufferIndex()].first = sizeof(int64_t);
                bufDim[bufid.getBufferIndex()].second[size-1] = 1;
            }
        }
        submitHandle->setBufferDimensions(bufDim);

        // *** DECODE BUFFER CREATION ***
        auto inputIdBufferDecode = createBuffer("input_ids", bufferMappings, true);
        auto positionIdBufferDecode = createBuffer("position_ids", bufferMappings, true);

        std::vector<std::vector<int64_t>> generated_ids(batch_size);
        std::vector<std::vector<int64_t>> logits(batch_size);
        std::vector<std::vector<int64_t>> position_id_for_decode(batch_size);

        // Total size of logits generated in outputBuffers
        int size_of_logits = outputBuffers[outputBuffers.size() - 1].size / (sizeof(float) *batch_size);

        // *** DECODE LOOP ***
        std::chrono::duration<double> elapsedDecode(0);
        for (int num_tokens = 1; num_tokens < generation_len.value() ; num_tokens++)
        {
            // Get data from outputBuffers into logits array
            get_logits_from_output_buffers(outputBuffers, logits, generated_ids, batch_size, size_of_logits);

            // Incrementing position ids by +1
            for(int bs=0;bs<batch_size;bs++){
                position_id_for_decode[bs] = {max_input_len_value_array[bs] + num_tokens};
            }

            auto startDecode = std::chrono::high_resolution_clock::now();
            // *** POPULATE DECODE BUFFERS ***
            populateBuffer(inputIdBufferDecode->getQBuffer(), logits);
            populateBuffer(positionIdBufferDecode->getQBuffer(), position_id_for_decode);

            // Fill last 2 index of inputBuffers with inputIds and positionIds
            inputBuffers[inputBuffers.size() - 1] = positionIdBufferDecode->getQBuffer();
            inputBuffers[inputBuffers.size() - 2] = inputIdBufferDecode->getQBuffer();

            submitHandle->setInputBuffers(inputBuffers);
            submitHandle->setOutputBuffers(outputBuffers);

            // *** SUBMIT ***
            status = inferenceSet->submit(submitHandle);
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in submitting handle through InferenceSet\n";
                return -1;
            }

            // *** COMPLETION ***
            status = inferenceSet->getCompleted(completedHandle);
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in getting completed handle through InferenceSet\n";
                return -1;
            }
            status = inferenceSet->putCompleted(std::move(completedHandle));
            if (status != QS_SUCCESS)
            {
                std::cerr << "Error in putting completed handle through InferenceSet\n";
                return -1;
            }
            auto endDecode = std::chrono::high_resolution_clock::now();
            elapsedDecode += (endDecode - startDecode);
        } //TODO: Add EOS

        // Filling last generated_ids from outputBuffers
        get_logits_from_output_buffers(outputBuffers, logits,generated_ids, batch_size, size_of_logits);
        int totalGeneratedIds = 0;
        for(ssize_t i=0;i<(int)generated_ids.size();i++){
            totalGeneratedIds += generated_ids[i].size();
        }

        std::cout<<"========================= Performance Stats =========================\n";
        std::chrono::duration<double> elapsedPrefill = prefillEnd - startPrefill;
        std::cout << "Prefill time a.k.a TTFT is= " << (elapsedPrefill.count()) << "\n";
        std::cout << "Decode Tokens/sec is= " << ((totalGeneratedIds-generated_ids.size())/elapsedDecode.count()) << "\n";
        std::chrono::duration<double> elapsedTotal = elapsedDecode + elapsedPrefill;
        std::cout << "Total Tokens/sec is= " << ((totalGeneratedIds)/elapsedTotal.count()) << "\n";
        std::cout << "Total (E2E) inference time is= " << (elapsedTotal.count()) << "\n";
        std::cout<<"=====================================================================\n";

        // Sending Generated Ids to Python to Generated Text using Tokenizer
        text_generation_inference.attr("tokenize_decode_output")(tokenizer, generated_ids, prompt).cast<py::array>();
    }

    catch (const py::error_already_set &e)
    {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
    return 0;
}

PYBIND11_MODULE(InferenceSetIOBuffer, m)
{
    m.doc() = "Running PyBind11";

    m.def("generatePrompt", &generatePrompt, "generatePrompt function");
}