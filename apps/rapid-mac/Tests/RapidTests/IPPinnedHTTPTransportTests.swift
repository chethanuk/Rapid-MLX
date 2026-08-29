import Foundation
import Testing
@testable import Rapid

@Suite("IP-pinned browse transport")
struct IPPinnedHTTPTransportTests {
    @Test("Chunked HTTP/1.1 bodies are decoded once")
    func chunkedResponseBodiesAreDecoded() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))
        let raw = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n4\r\n,123\r\n0\r\n\r\n".utf8)

        let (body, response) = try IPHTTPResponseParser.parse(data: raw, url: url)

        #expect(String(decoding: body, as: UTF8.self) == "abc,123")
        #expect(response.statusCode == 200)
        #expect(response.value(forHTTPHeaderField: "Transfer-Encoding") == "chunked")
    }

    @Test("A response without a complete header block fails closed")
    func incompleteHeaderResponseFailsClosed() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))

        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: Data("HTTP/1.1 200 OK\r\n".utf8), url: url)
        }
    }
}
